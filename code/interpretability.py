import os
import argparse
import random
import math
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.amp import autocast
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from model import MultimodalEnzymeModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def smart_load_weights(model, state_dict, model_name="Model"):
    model_keys = set(model.state_dict().keys())
    new_state_dict = {}
    loaded_keys = 0
    skipped_keys = 0

    print(f"🔧 [{model_name}] smart loading...")
    source_map = {}
    for k, v in state_dict.items():
        if "lora_" in k or "lora_A" in k or "lora_B" in k:
            continue
        clean_k = k.replace("base_model.model.", "")
        clean_k = clean_k.replace("module.", "")
        clean_k = clean_k.replace(".base_layer", "")
        source_map[clean_k] = v

    current_state = model.state_dict()
    for target_k in model_keys:
        if target_k in source_map and current_state[target_k].shape == source_map[target_k].shape:
            new_state_dict[target_k] = source_map[target_k]
            loaded_keys += 1
        else:
            skipped_keys += 1

    model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ [{model_name}] matched {loaded_keys}/{len(model_keys)} keys, skipped {skipped_keys}")
    return model


def get_batch_structures(seq_ids, max_len, structure_dict, device):
    batch_size = len(seq_ids)
    coords_batch = torch.zeros(batch_size, max_len, 3, dtype=torch.float32)
    mask_batch = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i, seq_id in enumerate(seq_ids):
        sid = str(seq_id)
        if sid in structure_dict:
            raw = structure_dict[sid]
            if isinstance(raw, np.ndarray):
                raw = torch.tensor(raw, dtype=torch.float32)
            else:
                raw = raw.float()
            valid_len = min(raw.shape[0], max_len - 2)
            if valid_len > 0:
                coords_batch[i, 1:valid_len + 1, :] = raw[:valid_len, :]
                mask_batch[i, 1:valid_len + 1] = 1.0
    return coords_batch.to(device), mask_batch.to(device)


def normalize_per_sample(x, mask, eps=1e-12):
    x = x * mask
    denom = x.sum(dim=1, keepdim=True).clamp_min(eps)
    return x / denom


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


@dataclass
class ExplainBatchOutput:
    pred_topt: np.ndarray
    fused_feat: torch.Tensor
    mask: torch.Tensor
    coords: torch.Tensor
    importance: torch.Tensor
    attention: torch.Tensor
    feature_norm: torch.Tensor


class TGCNetExplainer:
    def __init__(self, model_path, structure_path, esm_name, device="cuda", max_length=800):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        print(f"🚀 Device: {self.device}")
        print(f"📦 Loading structure dictionary: {structure_path}")
        self.structure_dict = safe_torch_load(structure_path, map_location="cpu")

        print(f"📦 Loading ESM-2: {esm_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(esm_name)
        self.plm_model = AutoModel.from_pretrained(esm_name).to(self.device)
        self.model = MultimodalEnzymeModel(dim=self.plm_model.config.hidden_size).to(self.device)

        print(f"📦 Loading checkpoint: {model_path}")
        checkpoint = safe_torch_load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if "plm_model_state_dict" in checkpoint:
                smart_load_weights(self.plm_model, checkpoint["plm_model_state_dict"], "ESM-2")
            else:
                warnings.warn("checkpoint has no plm_model_state_dict")

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print("✅ [TGC-Net] model_state_dict loaded")
            else:
                warnings.warn("checkpoint has no model_state_dict")
        else:
            self.model.load_state_dict(checkpoint, strict=False)
            print("✅ [TGC-Net] legacy checkpoint loaded")

        self.plm_model.eval()
        self.model.eval()

    def _encode_batch(self, seqs):
        return self.tokenizer(
            list(seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

    def _forward_to_fused(self, input_ids, seq_emb, coords, mask):
        model = self.model
        cnn_feat = model.local_cnn(input_ids)
        gate = model.fusion_gate(torch.cat([seq_emb, cnn_feat], dim=-1))
        fused_seq_emb = seq_emb + gate * cnn_feat

        dist_matrix = torch.cdist(coords, coords, p=2) / 10.0
        spatial_mask = (dist_matrix < 1.5).float()
        if mask is not None:
            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
            spatial_mask = spatial_mask * mask_2d

        struct_feat = fused_seq_emb
        for layer in model.struct_encoder:
            struct_feat = layer(struct_feat, dist_matrix, spatial_mask)

        fused_feat = model.fusion(fused_seq_emb, struct_feat, dist_matrix, mask)
        return fused_feat

    def _predict_from_fused(self, fused_feat, mask):
        model = self.model
        global_feat = model.pooler(fused_feat, mask)
        x = global_feat
        for rd in model.rds:
            x = rd(x)
        preds_norm = model.output_head(x).squeeze(-1)
        preds_topt = preds_norm * 120.0
        preds_topt = preds_topt.clamp(0.0, 120.0)
        return preds_topt

    def _importance_from_fused(self, fused_feat, mask):
        pooler = self.model.pooler
        keys = pooler.key_layer(fused_feat)
        scores = torch.matmul(keys, pooler.query.transpose(1, 2)).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        attention = F.softmax(scores, dim=1)
        gated_x = fused_feat * pooler.gate(fused_feat)
        feature_norm = torch.norm(gated_x, p=2, dim=-1)
        raw_importance = attention * feature_norm
        importance = normalize_per_sample(raw_importance, mask)
        return importance, attention * mask, feature_norm * mask

    @torch.no_grad()
    def explain_batch(self, ids, seqs):
        inputs = self._encode_batch(seqs)
        coords, mask = get_batch_structures(
            ids,
            inputs["input_ids"].shape[1],
            self.structure_dict,
            self.device
        )
        with autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            seq_emb = self.plm_model(**inputs).last_hidden_state
            fused_feat = self._forward_to_fused(inputs["input_ids"], seq_emb, coords, mask)
            pred_topt = self._predict_from_fused(fused_feat, mask)
            importance, attention, feature_norm = self._importance_from_fused(fused_feat, mask)
        return ExplainBatchOutput(
            pred_topt=pred_topt.float().detach().cpu().numpy(),
            fused_feat=fused_feat.float().detach(),
            mask=mask.float().detach(),
            coords=coords.float().detach(),
            importance=importance.float().detach(),
            attention=attention.float().detach(),
            feature_norm=feature_norm.float().detach(),
        )

    @torch.no_grad()
    def predict_after_masking(self, fused_feat_1, mask_1, positions, fill_value="zero"):
        fused_mod = fused_feat_1.clone()
        if len(positions) > 0:
            if fill_value == "zero":
                fused_mod[:, positions, :] = 0.0
            elif fill_value == "mean":
                valid = mask_1[0].bool()
                mean_vec = fused_mod[:, valid, :].mean(dim=1, keepdim=True)
                fused_mod[:, positions, :] = mean_vec
            else:
                raise ValueError("fill_value must be zero or mean")
        pred = self._predict_from_fused(fused_mod, mask_1)
        return float(pred.item())


def choose_case_ids(summary_df, n_cases=3):
    selected = []
    bins = [
        ("low", summary_df[summary_df["true_topt"] < 45]),
        ("middle", summary_df[(summary_df["true_topt"] >= 45) & (summary_df["true_topt"] < 70)]),
        ("high", summary_df[summary_df["true_topt"] >= 70]),
    ]
    for _, sub in bins:
        if len(sub) > 0:
            row = sub.sort_values("abs_error", ascending=True).iloc[0]
            selected.append(str(row["uniprot_id"]))
    if len(selected) < n_cases:
        rest = summary_df[~summary_df["uniprot_id"].astype(str).isin(selected)]
        add = rest.sort_values("abs_error", ascending=True).head(n_cases - len(selected))
        selected.extend(add["uniprot_id"].astype(str).tolist())
    return selected[:n_cases]


def run_interpretability(args):
    set_seed(args.seed)
    ensure_dir(args.outdir)

    explainer = TGCNetExplainer(
        model_path=args.model_path,
        structure_path=args.structure_path,
        esm_name=args.esm_name,
        device=args.device,
        max_length=args.max_length
    )

    df = pd.read_csv(args.input)
    required_cols = ["uniprot_id", "sequence", args.task]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    df["uniprot_id"] = df["uniprot_id"].astype(str)
    available = set(str(k) for k in explainer.structure_dict.keys())
    df = df[df["uniprot_id"].isin(available)].reset_index(drop=True)

    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No samples remain after matching uniprot_id with structure_dict.")

    avg_plddt_map = {}
    if args.quality_path and os.path.exists(args.quality_path):
        qdf = pd.read_csv(args.quality_path)
        if "Filename" in qdf.columns and "Avg_pLDDT" in qdf.columns:
            qdf["uniprot_id"] = qdf["Filename"].astype(str).str.replace(".pdb", "", regex=False)
            avg_plddt_map = dict(zip(qdf["uniprot_id"].astype(str), qdf["Avg_pLDDT"]))

    residue_rows = []
    sample_rows = []
    perturb_rows = []

    print(f"🧪 Running interpretability on {len(df)} samples...")

    for start in tqdm(range(0, len(df), args.batch_size), desc="Explain"):
        batch = df.iloc[start:start + args.batch_size].copy()
        ids = batch["uniprot_id"].astype(str).tolist()
        seqs = batch["sequence"].astype(str).tolist()
        out = explainer.explain_batch(ids, seqs)

        for bi, sid in enumerate(ids):
            seq = seqs[bi]
            true_topt = float(batch.iloc[bi][args.task])
            pred_topt = float(out.pred_topt[bi])
            abs_error = abs(pred_topt - true_topt)

            mask_np = out.mask[bi].detach().cpu().numpy()
            valid_token_positions = np.where(mask_np > 0.5)[0].astype(int)
            if len(valid_token_positions) == 0:
                continue

            imp_np = out.importance[bi].detach().cpu().numpy()
            attn_np = out.attention[bi].detach().cpu().numpy()
            norm_np = out.feature_norm[bi].detach().cpu().numpy()
            coords_np = out.coords[bi].detach().cpu().numpy()

            valid_importance = imp_np[valid_token_positions]
            k_top = max(1, int(math.ceil(len(valid_token_positions) * args.top_ratio)))
            sorted_valid = valid_token_positions[np.argsort(imp_np[valid_token_positions])[::-1]]
            top_positions = sorted_valid[:k_top]
            low_positions = sorted_valid[-k_top:]
            top_set = set(int(x) for x in top_positions)

            top_residue_labels = []
            for p in top_positions[:10]:
                residue_pos = int(p)
                aa = seq[residue_pos - 1] if 0 <= residue_pos - 1 < len(seq) else "X"
                top_residue_labels.append(f"{aa}{residue_pos}")

            for p in valid_token_positions:
                residue_pos = int(p)
                aa = seq[residue_pos - 1] if 0 <= residue_pos - 1 < len(seq) else "X"
                residue_rows.append({
                    "uniprot_id": sid,
                    "true_topt": true_topt,
                    "pred_topt": pred_topt,
                    "abs_error": abs_error,
                    "residue_position": residue_pos,
                    "aa": aa,
                    "importance": float(imp_np[p]),
                    "attention_weight": float(attn_np[p]),
                    "gated_feature_norm": float(norm_np[p]),
                    "x": float(coords_np[p, 0]),
                    "y": float(coords_np[p, 1]),
                    "z": float(coords_np[p, 2]),
                    "is_top_residue": int(int(p) in top_set),
                    "Avg_pLDDT": avg_plddt_map.get(sid, np.nan),
                })

            sample_rows.append({
                "uniprot_id": sid,
                "sequence_length": len(seq),
                "valid_structure_len": len(valid_token_positions),
                "true_topt": true_topt,
                "pred_topt": pred_topt,
                "abs_error": abs_error,
                "Avg_pLDDT": avg_plddt_map.get(sid, np.nan),
                "top_ratio": args.top_ratio,
                "top_k": k_top,
                "top_residues": ";".join(top_residue_labels),
                "importance_max": float(valid_importance.max()),
                "importance_entropy": float(-np.sum(valid_importance * np.log(valid_importance + 1e-12))),
            })

            fused_1 = out.fused_feat[bi:bi + 1].to(explainer.device)
            mask_1 = out.mask[bi:bi + 1].to(explainer.device)
            pred_before = explainer.predict_after_masking(fused_1, mask_1, [], fill_value=args.fill_value)

            k_mask = max(1, int(math.ceil(len(valid_token_positions) * args.mask_ratio)))
            top_mask_positions = sorted_valid[:k_mask].tolist()
            low_mask_positions = sorted_valid[-k_mask:].tolist()

            strategies = {
                "Top important": [(top_mask_positions, 0)],
                "Low important": [(low_mask_positions, 0)],
            }

            rng = np.random.default_rng(args.seed + start + bi)
            random_sets = []
            for rr in range(args.random_repeats):
                random_positions = rng.choice(valid_token_positions, size=k_mask, replace=False).astype(int).tolist()
                random_sets.append((random_positions, rr))
            strategies["Random"] = random_sets

            for strategy, pos_repeat_pairs in strategies.items():
                for positions, repeat_idx in pos_repeat_pairs:
                    pred_after = explainer.predict_after_masking(fused_1, mask_1, positions, fill_value=args.fill_value)
                    perturb_rows.append({
                        "uniprot_id": sid,
                        "true_topt": true_topt,
                        "pred_before": pred_before,
                        "strategy": strategy,
                        "repeat": repeat_idx,
                        "mask_ratio": args.mask_ratio,
                        "k_mask": k_mask,
                        "pred_after": pred_after,
                        "delta_pred": pred_after - pred_before,
                        "abs_delta_pred": abs(pred_after - pred_before),
                        "masked_token_positions": ";".join(str(int(x)) for x in positions),
                    })

    residue_df = pd.DataFrame(residue_rows)
    summary_df = pd.DataFrame(sample_rows)
    perturb_df = pd.DataFrame(perturb_rows)

    residue_path = os.path.join(args.outdir, "residue_importance_long.csv")
    summary_path = os.path.join(args.outdir, "sample_summary.csv")
    perturb_path = os.path.join(args.outdir, "perturbation_results.csv")
    perturb_summary_path = os.path.join(args.outdir, "perturbation_summary.csv")

    residue_df.to_csv(residue_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    perturb_df.to_csv(perturb_path, index=False)

    perturb_summary = perturb_df.groupby("strategy")["abs_delta_pred"].agg(["count", "mean", "median", "std"]).reset_index()
    perturb_summary.to_csv(perturb_summary_path, index=False)

    print(f"Saved: {residue_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {perturb_path}")
    print(f"Saved: {perturb_summary_path}")

    metrics = compute_metrics(summary_df["true_topt"], summary_df["pred_topt"])
    print("\nPrediction metrics on interpreted samples:")
    print(f"R2   = {metrics['R2']:.4f}")
    print(f"RMSE = {metrics['RMSE']:.2f} °C")
    print(f"MAE  = {metrics['MAE']:.2f} °C")

    if args.case_ids:
        case_ids = [x.strip() for x in args.case_ids.split(",") if x.strip()]
    else:
        case_ids = choose_case_ids(summary_df, n_cases=3)

    case_ids = [cid for cid in case_ids if cid in set(summary_df["uniprot_id"].astype(str))]
    if len(case_ids) == 0:
        case_ids = summary_df.sort_values("abs_error", ascending=True).head(3)["uniprot_id"].astype(str).tolist()

    structure_case_id = args.structure_case_id
    if structure_case_id is None or structure_case_id not in set(summary_df["uniprot_id"].astype(str)):
        structure_case_id = case_ids[-1]

    plot_interpretability_figures(
        residue_df=residue_df,
        summary_df=summary_df,
        perturb_df=perturb_df,
        case_ids=case_ids,
        structure_case_id=structure_case_id,
        outdir=args.outdir,
        top_ratio=args.top_ratio
    )


def setup_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def add_panel_label(ax, label, x=-0.12, y=1.10):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=22, fontweight="bold", ha="left", va="top", color="#111111", clip_on=False)


def save_fig(fig, path_prefix):
    fig.savefig(path_prefix + ".png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(path_prefix + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(path_prefix + ".svg", bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {path_prefix}.png/pdf/svg")


def pca_2d(coords):
    coords = np.asarray(coords, dtype=float)
    coords = coords - coords.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(coords, full_matrices=False)
    return coords @ vt[:2].T



def plot_importance_profiles(residue_df, summary_df, case_ids, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.2))
    else:
        fig = ax.figure

    line_handles = []
    line_labels = []
    max_pos = 0

    for sid in case_ids:
        sub = residue_df[residue_df["uniprot_id"].astype(str) == str(sid)].copy()
        if len(sub) == 0:
            continue

        max_pos = max(max_pos, int(sub["residue_position"].max()))
        info = summary_df[summary_df["uniprot_id"].astype(str) == str(sid)].iloc[0]
        label = f"{sid}  True={info['true_topt']:.0f}°C, Pred={info['pred_topt']:.1f}°C"

        (line,) = ax.plot(
            sub["residue_position"],
            sub["importance"],
            linewidth=1.55,
            alpha=0.95,
            label=label
        )
        line_handles.append(line)
        line_labels.append(label)

        top = sub[sub["is_top_residue"] == 1].copy()
        ax.scatter(
            top["residue_position"],
            top["importance"],
            s=34,
            facecolor=line.get_color(),
            edgecolor="#222222",
            linewidth=0.55,
            alpha=0.95,
            zorder=4
        )

    ax.set_xlabel("Residue position")
    ax.set_ylabel("Residue importance")
    ax.set_title("Residue-level importance profiles")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.75)
    ax.legend(line_handles, line_labels, fontsize=8.4, frameon=True, loc="upper right")

    inset = ax.inset_axes([0.055, 0.55, 0.30, 0.36])
    eps = 1e-8
    for sid in case_ids:
        sub = residue_df[residue_df["uniprot_id"].astype(str) == str(sid)].copy()
        if len(sub) == 0:
            continue
        inset.plot(
            sub["residue_position"],
            np.log10(sub["importance"].values + eps),
            linewidth=1.0,
            alpha=0.9
        )

    inset.set_title("log-scale view", fontsize=8.2, pad=3)
    inset.set_xlabel("Position", fontsize=7.5)
    inset.set_ylabel("log10 importance", fontsize=7.5)
    inset.tick_params(axis="both", labelsize=7)
    inset.grid(True, linestyle=":", linewidth=0.6, alpha=0.65)
    for spine in inset.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")

    if max_pos > 0:
        ax.set_xlim(0, max_pos + max(10, int(max_pos * 0.03)))

    return fig, ax


def plot_structure_projection(residue_df, summary_df, structure_case_id, top_ratio, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 5.0))
    else:
        fig = ax.figure

    sub = residue_df[residue_df["uniprot_id"].astype(str) == str(structure_case_id)].copy()
    if len(sub) == 0:
        ax.text(0.5, 0.5, "No structure case available", ha="center", va="center")
        return fig, ax

    coords = sub[["x", "y", "z"]].values
    xy = pca_2d(coords)
    sub["pc1"] = xy[:, 0]
    sub["pc2"] = xy[:, 1]

    top = sub[sub["is_top_residue"] == 1].copy()

    ax.scatter(
        sub["pc1"], sub["pc2"],
        s=22,
        color="#CFCFCF",
        edgecolor="none",
        alpha=0.62,
        label="Other residues",
        zorder=1
    )

    if len(top) > 0:
        imp = top["importance"].values.astype(float)
        imp_min, imp_max = imp.min(), imp.max()
        if imp_max > imp_min:
            sizes = 46 + 130 * (imp - imp_min) / (imp_max - imp_min)
        else:
            sizes = np.full_like(imp, 80.0)

        ax.scatter(
            top["pc1"], top["pc2"],
            s=sizes,
            facecolor="#C44E52",
            edgecolor="#111111",
            linewidth=0.65,
            alpha=0.95,
            label=f"Top {int(top_ratio * 100)}% residues",
            zorder=4
        )

    top8 = top.sort_values("importance", ascending=False).head(8)
    for _, row in top8.iterrows():
        ax.text(
            row["pc1"], row["pc2"],
            f"{row['aa']}{int(row['residue_position'])}",
            fontsize=8.7,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="#111111",
            bbox=dict(
                boxstyle="round,pad=0.14",
                facecolor="white",
                edgecolor="#DDDDDD",
                linewidth=0.35,
                alpha=0.88
            ),
            zorder=6
        )

    info = summary_df[summary_df["uniprot_id"].astype(str) == str(structure_case_id)].iloc[0]
    ax.set_title(
        f"Structure projection of {structure_case_id}"
        f"True={info['true_topt']:.0f}°C, Pred={info['pred_topt']:.1f}°C",
        fontsize=11.5
    )
    ax.set_xlabel("PC1 of Cα coordinates")
    ax.set_ylabel("PC2 of Cα coordinates")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.75)
    ax.legend(fontsize=8.5, frameon=True, loc="best")

    ax.text(
        0.98, 0.02,
        "Red point size indicates importance",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#DDDDDD", alpha=0.85)
    )

    return fig, ax

def plot_perturbation_box(perturb_df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.8, 4.8))
    else:
        fig = ax.figure

    order = ["Low important", "Random", "Top important"]
    colors = {"Low important": "#B0B0B0", "Random": "#4C78A8", "Top important": "#C44E52"}
    data = [perturb_df[perturb_df["strategy"] == s]["abs_delta_pred"].values for s in order]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=order, showfliers=False, widths=0.58)

    for patch, s in zip(bp["boxes"], order):
        patch.set_facecolor(colors[s])
        patch.set_alpha(0.75)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(1.2)
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("#222222")
            item.set_linewidth(1.2)

    rng = np.random.default_rng(42)
    for i, s in enumerate(order, start=1):
        vals = perturb_df[perturb_df["strategy"] == s]["abs_delta_pred"].values
        if len(vals) == 0:
            continue
        x = rng.normal(i, 0.055, size=len(vals))
        ax.scatter(x, vals, s=18, color="#222222", alpha=0.35, linewidth=0)

    ax.set_ylabel("|Δ predicted Topt| (°C)")
    ax.set_title("Feature perturbation validation")
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.75)
    return fig, ax


def plot_interpretability_figures(residue_df, summary_df, perturb_df, case_ids, structure_case_id, outdir, top_ratio):
    setup_plot_style()

    fig_a, ax_a = plt.subplots(figsize=(10.8, 4.8))
    plot_importance_profiles(residue_df, summary_df, case_ids, ax=ax_a)
    add_panel_label(ax_a, "A")
    fig_a.tight_layout()
    save_fig(fig_a, os.path.join(outdir, "figure9A_importance_profiles"))
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(6.6, 5.8))
    plot_structure_projection(residue_df, summary_df, structure_case_id, top_ratio, ax=ax_b)
    add_panel_label(ax_b, "B")
    fig_b.tight_layout()
    save_fig(fig_b, os.path.join(outdir, "figure9B_structure_projection"))
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(6.4, 5.2))
    plot_perturbation_box(perturb_df, ax=ax_c)
    add_panel_label(ax_c, "C")
    fig_c.tight_layout()
    save_fig(fig_c, os.path.join(outdir, "figure9C_perturbation_validation"))
    plt.close(fig_c)

    fig = plt.figure(figsize=(16.5, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.05], width_ratios=[1.38, 1.0], hspace=0.34, wspace=0.26)
    ax1 = fig.add_subplot(gs[0, :])
    plot_importance_profiles(residue_df, summary_df, case_ids, ax=ax1)
    add_panel_label(ax1, "A", x=-0.055, y=1.10)
    ax2 = fig.add_subplot(gs[1, 0])
    plot_structure_projection(residue_df, summary_df, structure_case_id, top_ratio, ax=ax2)
    add_panel_label(ax2, "B", x=-0.075, y=1.10)
    ax3 = fig.add_subplot(gs[1, 1])
    plot_perturbation_box(perturb_df, ax=ax3)
    add_panel_label(ax3, "C", x=-0.12, y=1.10)
    fig.suptitle("Residue-level interpretability analysis of TGC-Net", fontsize=17, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_fig(fig, os.path.join(outdir, "figure9_interpretability"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/ys/new/TGC-Net/data/topt/test/test.csv")
    parser.add_argument("--model_path", default="/home/ys/new/TGC-Net/pLDDT/0.7topt_best_model_0.624.pth")
    parser.add_argument("--structure_path", default="/home/ys/new/TGC-Net/data/topt/test/291.pt")
    parser.add_argument("--quality_path", default="/home/ys/new/TGC-Net/data/topt/test/291-quality.csv")
    parser.add_argument("--outdir", default="./interpretability_results")
    parser.add_argument("--esm_name", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--task", default="topt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_length", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--top_ratio", type=float, default=0.10, help="Top residue ratio for visualization.")
    parser.add_argument("--mask_ratio", type=float, default=0.10, help="Residue ratio to mask in perturbation.")
    parser.add_argument("--random_repeats", type=int, default=5)
    parser.add_argument("--fill_value", choices=["zero", "mean"], default="zero")
    parser.add_argument("--case_ids", default=None, help="Comma-separated UniProt IDs for panel A.")
    parser.add_argument("--structure_case_id", default=None, help="One UniProt ID for panel B.")
    parser.add_argument("--max_samples", type=int, default=0, help="Use a subset for debugging. 0 means all samples.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_interpretability(args)


if __name__ == "__main__":
    main()