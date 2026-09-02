import os
import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix
from torch.amp import autocast
from model import MultimodalEnzymeModel


def get_batch_structures(seq_ids, max_len, structure_dict, device):
    batch_size = len(seq_ids)
    coords_batch = torch.zeros(batch_size, max_len, 3)
    mask_batch = torch.zeros(batch_size, max_len)
    for i, seq_id in enumerate(seq_ids):
        sid = str(seq_id)
        if sid in structure_dict:
            raw = structure_dict[sid]
            valid_len = min(raw.shape[0], max_len - 2)
            if valid_len > 0:
                coords_batch[i, 1:valid_len + 1, :] = raw[:valid_len, :]
                mask_batch[i, 1:valid_len + 1] = 1.0
    return coords_batch.to(device), mask_batch.to(device)


def smart_load_weights(model, state_dict, model_name="Model"):
    model_keys = set(model.state_dict().keys())
    new_state_dict = {}
    loaded_keys = 0
    skipped_keys = 0
    source_map = {}
    for k, v in state_dict.items():
        if "lora_" in k or "lora_A" in k or "lora_B" in k:
            continue
        clean_k = k.replace("base_model.model.", "").replace("module.", "").replace(".base_layer", "")
        source_map[clean_k] = v
    for target_k in model_keys:
        if target_k in source_map:
            if model.state_dict()[target_k].shape == source_map[target_k].shape:
                new_state_dict[target_k] = source_map[target_k]
                loaded_keys += 1
            else:
                skipped_keys += 1
        else:
            skipped_keys += 1
    model.load_state_dict(new_state_dict, strict=False)
    return True


class MultimodalPhPredictor:
    def __init__(self, model_path, esm_name, structure_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Initialize inference environment (Device: {self.device})")
        if not os.path.exists(structure_path):
            raise FileNotFoundError(f"Structural data not found: {structure_path}")
        self.structure_dict = torch.load(structure_path, map_location='cpu')
        print(f"Initialization standards ESM-2: {esm_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(esm_name)
        self.plm_model = AutoModel.from_pretrained(esm_name).to(self.device)
        self.model = MultimodalEnzymeModel(dim=self.plm_model.config.hidden_size).to(self.device)
        print(f"Loading checkpoints: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if 'plm_model_state_dict' in checkpoint:
                smart_load_weights(self.plm_model, checkpoint['plm_model_state_dict'], "ESM-2")
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        self.plm_model.eval()
        self.model.eval()

    def predict(self, df, batch_size=1):
        ids = df['uniprot_id'].astype(str).values
        seqs = df['sequence'].values
        all_preds = []
        min_v, max_v = 0.0, 14.0
        print(f"Start predicting {len(ids)} pieces of data...")
        with torch.no_grad():
            with tqdm(total=len(ids), desc="Predicting", unit="seq") as pbar:
                for i in range(0, len(ids), batch_size):
                    b_ids = ids[i:i + batch_size]
                    b_seqs = list(seqs[i:i + batch_size])
                    inputs = self.tokenizer(b_seqs, return_tensors="pt", padding=True,
                                            truncation=True, max_length=800).to(self.device)
                    coords, mask = get_batch_structures(b_ids, inputs['input_ids'].shape[1],
                                                        self.structure_dict, self.device)
                    with autocast(device_type=self.device.type):
                        seq_emb = self.plm_model(**inputs).last_hidden_state
                        preds_norm = self.model(inputs['input_ids'], seq_emb, coords, mask=mask).squeeze(-1)
                    batch_preds = preds_norm.float().cpu().numpy() * (max_v - min_v) + min_v
                    all_preds.extend(batch_preds)
                    pbar.update(len(b_ids))
        df['pred_pHopt'] = all_preds
        return df

    def paper_style_benchmark_visualization(self, df, save_path, task_col='pHopt'):
        if task_col not in df.columns:
            return
        y_true = df[task_col].values
        y_pred = df['pred_pHopt'].values
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print("\n" + "=" * 50)
        print(f" TGC-Net Final Test Results:")
        print(f"   R2:   {r2:.3f}")
        print(f"   MAE:  {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print("=" * 50 + "\n")
        sns.set_theme(style="ticks")
        plt.figure(figsize=(7, 6))

        sns.scatterplot(
            x=y_true,
            y=y_pred,
            alpha=0.75,
            color='#9467bd',
            edgecolor='w',
            s=60,
            label=f'Predicted Samples (n={len(y_true)})'
        )

        all_values = np.concatenate([y_true, y_pred])
        min_val, max_val = all_values.min() - 5, all_values.max() + 5
        plt.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', lw=2, label='Perfect Match (y=x)')

        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        stats_text = (
            f"$R^2 = {r2:.3f}$\n"
            f"RMSE = {rmse:.2f}\n"
            f"MAE = {mae:.2f}"
        )
        plt.text(
            0.05, 0.95, stats_text,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#cccccc')
        )

        plt.xlabel('Experimental pHopt', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted pHopt', fontsize=12, fontweight='bold')

        plt.title("TGC-Net (Proposed) - Evaluation on Test Set", fontsize=13, fontweight='bold', pad=15)
        plt.legend(loc='lower right', frameon=True)
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"TGC-Net benchmark comparison chart has been saved to: {save_path}")
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/home/ys/new/TGC-Net/data/ph/test-ph.csv')
    parser.add_argument('--output', default='/home/ys/new/TGC-Net/data/result/ph/result1.csv')
    parser.add_argument('--model_path', default='/home/ys/new/TGC-Net/data/result/ph/ph_best_model.pth')
    parser.add_argument('--structure_path', default='/home/ys/new/TGC-Net/data/ph/ph-structures.pt')
    parser.add_argument('--quality_path', default='/home/ys/new/TGC-Net/data/ph/ph-quality.csv')
    parser.add_argument('--esm_name', default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    predictor = MultimodalPhPredictor(args.model_path, args.esm_name, args.structure_path)

    df = pd.read_csv(args.input)
    print(f"Original quantity: {len(df)}")

    df['uniprot_id'] = df['uniprot_id'].astype(str)
    df = df[df['uniprot_id'].isin(predictor.structure_dict.keys())].reset_index(drop=True)

    if os.path.exists(args.quality_path):
        qual_df = pd.read_csv(args.quality_path)
        qual_df['uniprot_id'] = qual_df['Filename'].str.replace('.pdb', '', regex=False)
        df = df.merge(qual_df[['uniprot_id', 'Avg_pLDDT']], on='uniprot_id', how='left')

    if len(df) == 0: return

    res_df = predictor.predict(df, batch_size=args.batch_size)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    res_df.to_csv(args.output, index=False)

    vis_path = args.output.replace('.csv', 'TGC.png')
    predictor.paper_style_benchmark_visualization(res_df, vis_path)


if __name__ == "__main__":
    main()