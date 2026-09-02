import os
import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from peft import get_peft_model, LoraConfig, TaskType
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
        clean_k = k.replace("base_model.model.", "").replace("module.", "")
        clean_k = clean_k.replace(".base_layer", "")
        source_map[clean_k] = v
    for target_k in model_keys:
        if target_k in source_map and model.state_dict()[target_k].shape == source_map[target_k].shape:
            new_state_dict[target_k] = source_map[target_k]
            loaded_keys += 1
        else:
            skipped_keys += 1
    model.load_state_dict(new_state_dict, strict=False)
    return model


def plot_prediction_results(y_true, y_pred, save_path="/home/ys/new/TGC-Net/data/result/tm/test_prediction.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 8))
    sns.set_style("whitegrid")
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='#1f77b4')
    limit = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    plt.plot(limit, limit, color='#d62728', lw=2, linestyle='--', label='Perfect fit')
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.title(f'Final Independent Test\nR2: {r2:.3f} | MAE: {mae:.2f}°C | RMSE: {rmse:.2f}°C')
    plt.xlabel('Experimental Tm (°C)')
    plt.ylabel('Predicted Tm (°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_history(history, save_path="/home/ys/new/TGC-Net/data/result/tm/training.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(epochs, history['train_loss'], 'b-', marker='o', label='Train Loss')
    ax1.set_title('Training Loss per Epoch')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    ax2.plot(epochs, history['dev_rmse'], 'r-', marker='s', label='Validation RMSE')
    ax2.plot(epochs, history['dev_mae'], color='orange', linestyle='-', marker='d', label='Validation MAE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Error (°C)')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax3 = ax2.twinx()
    ax3.plot(epochs, history['dev_r2'], 'g--', marker='^', label='Validation R2')
    ax3.set_ylabel('R2 Score')
    ax3.legend(loc='upper right')
    ax2.set_title('Validation Performance Over Time')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def evaluate_model(model, plm_model, tokenizer, df, structure_dict, device, args, min_v, max_v, desc="Evaluating"):
    model.eval()
    plm_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        iterator = tqdm(range(0, len(df), args.batch_size), desc=desc, leave=False)
        for i in iterator:
            batch = df.iloc[i: i + args.batch_size]
            b_ids = batch['uniprot_id'].values
            b_seqs = list(batch['sequence'].values)
            b_y_norm = (batch[args.task].values - min_v) / (max_v - min_v)
            inputs = tokenizer(
                b_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length
            ).to(device)
            coords, mask = get_batch_structures(b_ids, inputs['input_ids'].shape[1], structure_dict, device)
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                seq_emb = plm_model(**inputs).last_hidden_state
                preds = model(inputs['input_ids'], seq_emb, coords, mask=mask).squeeze(-1)
            all_preds.extend(preds.float().cpu().numpy())
            all_labels.extend(b_y_norm)
    y_true = np.array(all_labels) * (max_v - min_v) + min_v
    y_pred = np.array(all_preds) * (max_v - min_v) + min_v
    y_pred = np.clip(y_pred, min_v, max_v)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse, y_true, y_pred


def save_resume_checkpoint(path, epoch, model, plm_model, optimizer, scheduler, scaler, history, best_dev_rmse, best_dev_r2, best_model_path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'plm_model_state_dict': plm_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'history': history,
        'best_dev_rmse': best_dev_rmse,
        'best_dev_r2': best_dev_r2,
        'best_model_path': best_model_path
    }
    torch.save(state, path)


def load_resume_checkpoint(path, model, plm_model, optimizer, scheduler, scaler):
    if os.path.exists(path):
        print(f"\nDiscovered breakpoint file: {path}，Restoring training progress...")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        plm_model.load_state_dict(checkpoint['plm_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', {
            'train_loss': [], 'dev_r2': [], 'dev_rmse': [], 'dev_mae': []
        })
        best_dev_rmse = checkpoint.get('best_dev_rmse', float('inf'))
        best_dev_r2 = checkpoint.get('best_dev_r2', -float('inf'))
        best_model_path = checkpoint.get('best_model_path', None)
        print(f" Epoch {start_epoch + 1} Continue to start")
        print(f" The current best Validation RMSE: {best_dev_rmse:.4f}")
        print(f" The current best Validation R2:   {best_dev_r2:.4f}\n")
        return start_epoch, history, best_dev_rmse, best_dev_r2, best_model_path
    default_history = {
        'train_loss': [],
        'dev_r2': [],
        'dev_rmse': [],
        'dev_mae': []
    }
    return 0, default_history, float('inf'), -float('inf'), None


def save_best_model_by_validation(save_path, epoch, model, plm_model, dev_r2, dev_mae, dev_rmse):
    base_dir = os.path.dirname(save_path)
    file_name = os.path.basename(save_path)
    name_no_ext, ext = os.path.splitext(file_name)
    current_save_path = os.path.join(
        base_dir,
        f"{name_no_ext}_bestDevRMSE_{dev_rmse:.4f}_devR2_{dev_r2:.4f}{ext}"
    )
    plm_model.merge_adapter()
    full_plm_state_dict = plm_model.state_dict()
    checkpoint = {
        'plm_model_state_dict': full_plm_state_dict,
        'model_state_dict': model.state_dict(),
        'best_dev_rmse': dev_rmse,
        'best_dev_r2': dev_r2,
        'best_dev_mae': dev_mae,
        'epoch': epoch,
        'selection_metric': 'validation_rmse'
    }
    torch.save(checkpoint, current_save_path)
    plm_model.unmerge_adapter()
    return current_save_path


def load_best_model_for_final_test(best_model_path, model_name, device):
    if best_model_path is None or not os.path.exists(best_model_path):
        raise FileNotFoundError(f"no find: {best_model_path}")
    print(f"\nLoad the best model from the validation set for the final independent testing: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    final_plm_model = AutoModel.from_pretrained(model_name).to(device)
    final_model = MultimodalEnzymeModel(dim=final_plm_model.config.hidden_size).to(device)
    smart_load_weights(final_plm_model, checkpoint['plm_model_state_dict'], "Merged ESM-2")
    final_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    final_plm_model.eval()
    final_model.eval()
    return final_model, final_plm_model, tokenizer


def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='/home/ys/new/TGC-Net/data/tm/train-tm.csv')
    parser.add_argument('--test_path', type=str, default='/home/ys/new/TGC-Net/data/tm/test-tm.csv')
    parser.add_argument('--structure_path', type=str, default='/home/ys/new/TGC-Net/data/tm/tm-structures.pt')
    parser.add_argument('--test_structure_path', type=str, default='/home/ys/new/TGC-Net/data/tm/tm-structures.pt')
    parser.add_argument('--quality_path', type=str, default='/home/ys/new/TGC-Net/data/tm/quality_report.csv')
    parser.add_argument('--plddt_threshold', type=float, default=0.7)
    parser.add_argument('--task', type=str, default='tm')
    parser.add_argument('--target_min', type=float, default=0.0)
    parser.add_argument('--target_max', type=float, default=100.0)
    parser.add_argument('--model_name', type=str, default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--max_length', type=int, default=800)
    parser.add_argument('--save_path', type=str, default='/home/ys/new/TGC-Net/data/result/tm/tm_best_model.pth')
    parser.add_argument('--ckpt_path', type=str, default='/home/ys/new/TGC-Net/data/result/tm/checkpoint.pth')
    parser.add_argument('--final_pred_path', type=str, default='/home/ys/new/TGC-Net/data/result/tm/results-tm.csv')
    parser.add_argument('--history_fig_path', type=str, default='/home/ys/new/TGC-Net/data/result/tm/training.png')
    parser.add_argument('--final_fig_path', type=str, default='/home/ys/new/TGC-Net/data/result/tm/test_prediction.png')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.final_pred_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading training/validation structure, independent test structure, and quality report ..")
    train_structure_dict = torch.load(
        args.structure_path,
        map_location='cpu',
        weights_only=False
    )
    test_structure_dict = torch.load(
        args.test_structure_path,
        map_location='cpu',
        weights_only=False
    )
    quality_df = pd.read_csv(args.quality_path)
    quality_df['uniprot_id'] = quality_df['Filename'].astype(str).str.replace('.pdb', '', regex=False)
    high_quality_ids = set(
        quality_df[quality_df['Avg_pLDDT'] >= args.plddt_threshold]['uniprot_id'].astype(str)
    )
    train_available_ids = set(str(k) for k in train_structure_dict.keys())
    test_available_ids = set(str(k) for k in test_structure_dict.keys())
    df_raw_train = pd.read_csv(args.train_path)
    df_te = pd.read_csv(args.test_path)
    required_cols = {'uniprot_id', 'sequence', args.task}
    missing_train = required_cols - set(df_raw_train.columns)
    missing_test = required_cols - set(df_te.columns)
    if missing_train:
        raise ValueError(f"The training set lacks essential columns: {missing_train}")
    if missing_test:
        raise ValueError(f"The test set lacks necessary columns: {missing_test}")
    df_raw_train['uniprot_id'] = df_raw_train['uniprot_id'].astype(str)
    df_te['uniprot_id'] = df_te['uniprot_id'].astype(str)
    df_raw_train = df_raw_train[
        df_raw_train['uniprot_id'].isin(high_quality_ids & train_available_ids)
    ].reset_index(drop=True)
    df_te = df_te[
        df_te['uniprot_id'].isin(test_available_ids)
    ].reset_index(drop=True)
    df_raw_train = df_raw_train.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_raw_train) * 0.9)
    df_tr = df_raw_train.iloc[:split_idx].reset_index(drop=True)
    df_dev = df_raw_train.iloc[split_idx:].reset_index(drop=True)
    print(f"Data is ready")
    train_targets = df_tr[args.task].values
    if args.task.lower() in ["topt", "optimal_temperature"]:
        sample_weights = np.zeros_like(train_targets, dtype=np.float32)
        for i, t in enumerate(train_targets):
            if t < 20.0:
                sample_weights[i] = 5.0
            elif t >= 80.0:
                sample_weights[i] = 3.0
            elif t >= 60.0:
                sample_weights[i] = 1.5
            else:
                sample_weights[i] = 1.0
    else:
        sample_weights = np.ones_like(train_targets, dtype=np.float32)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )
    train_dataset = TensorDataset(torch.arange(len(df_tr)))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, drop_last=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    plm_model = AutoModel.from_pretrained(args.model_name).to(device)
    print("Currently applying LoRA (r=16)...")
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value", "dense", "intermediate.dense"]
    )
    plm_model = get_peft_model(plm_model, peft_config)
    plm_model.print_trainable_parameters()
    model = MultimodalEnzymeModel(dim=plm_model.config.hidden_size).to(device)
    lora_params = filter(lambda p: p.requires_grad, plm_model.parameters())
    new_module_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW([
        {'params': lora_params, 'lr': args.lr},
        {'params': new_module_params, 'lr': args.lr * 2.0}
    ], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6
    )
    scaler = GradScaler(device='cuda', enabled=(device.type == "cuda"))
    criterion = nn.MSELoss()
    start_epoch, history, best_dev_rmse, best_dev_r2, best_model_path = load_resume_checkpoint(
        args.ckpt_path, model, plm_model, optimizer, scheduler, scaler
    )
    last_saved_path = best_model_path
    if start_epoch >= args.epochs:
        print("The model has completed all Epoch training and will be directly loaded with the best model from the validation set for final testing.")
    else:
        min_v, max_v = args.target_min, args.target_max
        for epoch in range(start_epoch, args.epochs):
            model.train()
            plm_model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} Train")
            epoch_losses = []
            optimizer.zero_grad()
            for i, (idx_tensor,) in enumerate(pbar):
                idx = idx_tensor.numpy()
                b_ids = df_tr['uniprot_id'].values[idx]
                b_seqs = list(df_tr['sequence'].values[idx])
                b_labels = torch.tensor(
                    (df_tr[args.task].values[idx] - min_v) / (max_v - min_v),
                    dtype=torch.float
                ).to(device)
                inputs = tokenizer(
                    b_seqs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length
                ).to(device)
                coords, mask = get_batch_structures(
                    b_ids,
                    inputs['input_ids'].shape[1],
                    train_structure_dict,
                    device
                )
                coords = coords + torch.randn_like(coords) * 0.05 * mask.unsqueeze(-1)
                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    seq_emb = plm_model(**inputs).last_hidden_state
                    preds = model(inputs['input_ids'], seq_emb, coords, mask=mask).squeeze(-1)
                    loss = criterion(preds.view(-1), b_labels.view(-1)) / args.accumulation_steps
                scaler.scale(loss).backward()
                epoch_losses.append(loss.item() * args.accumulation_steps)
                if (i + 1) % args.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(plm_model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                pbar.set_postfix({'loss': f"{epoch_losses[-1]:.4f}"})
            if len(train_loader) % args.accumulation_steps != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(plm_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            avg_train_loss = np.mean(epoch_losses)
            dev_r2, dev_mae, dev_rmse, _, _ = evaluate_model(
                model, plm_model, tokenizer, df_dev, train_structure_dict,
                device, args, min_v, max_v, desc="Eval Validation"
            )
            history['train_loss'].append(avg_train_loss)
            history['dev_r2'].append(dev_r2)
            history['dev_rmse'].append(dev_rmse)
            history['dev_mae'].append(dev_mae)
            print(f"⭐ Epoch {epoch + 1}")
            print(f"   Train Loss:      {avg_train_loss:.4f}")
            print(f"   Validation RMSE: {dev_rmse:.2f} | MAE: {dev_mae:.2f} | R2: {dev_r2:.4f}")
            scheduler.step(dev_rmse)
            if dev_rmse < best_dev_rmse:
                best_dev_rmse = dev_rmse
                best_dev_r2 = dev_r2
                current_save_path = save_best_model_by_validation(
                    args.save_path, epoch, model, plm_model, dev_r2, dev_mae, dev_rmse
                )
                print(f" Validation RMSE update！The best model has been saved to {current_save_path}")
                if last_saved_path and last_saved_path != current_save_path and os.path.exists(last_saved_path):
                    try:
                        os.remove(last_saved_path)
                        print(f"Old validation set best model deleted: {os.path.basename(last_saved_path)}")
                    except OSError:
                        pass
                last_saved_path = current_save_path
                best_model_path = current_save_path
            save_resume_checkpoint(
                args.ckpt_path, epoch, model, plm_model, optimizer, scheduler,
                scaler, history, best_dev_rmse, best_dev_r2, best_model_path
            )
        plot_training_history(history, args.history_fig_path)
    if best_model_path is None:
        raise RuntimeError("No validation set best model has been saved, and final testing cannot be conducted. Please review the training process.")
    try:
        del model
        del plm_model
        torch.cuda.empty_cache()
    except Exception:
        pass
    final_model, final_plm_model, final_tokenizer = load_best_model_for_final_test(
        best_model_path, args.model_name, device
    )
    final_r2, final_mae, final_rmse, y_true, y_pred = evaluate_model(
        final_model, final_plm_model, final_tokenizer, df_te, test_structure_dict,
        device, args, args.target_min, args.target_max, desc="Final Independent Test"
    )
    print("\n" + "=" * 60)
    print("🎯 Final Independent Test Results")
    print(f"   Selected by: Validation RMSE")
    print(f"   Best Validation RMSE: {best_dev_rmse:.4f}")
    print(f"   Final Test R2:        {final_r2:.4f}")
    print(f"   Final Test RMSE:      {final_rmse:.2f}")
    print(f"   Final Test MAE:       {final_mae:.2f}")
    print("=" * 60 + "\n")
    out_df = df_te.copy()
    out_df['true_tm'] = y_true
    out_df['pred_tm'] = y_pred
    out_df['abs_error'] = np.abs(y_pred - y_true)
    out_df.to_csv(args.final_pred_path, index=False)
    print(f" The final test prediction results have been saved: {args.final_pred_path}")
    plot_prediction_results(y_true, y_pred, args.final_fig_path)
    print(f" The final test scatter plot has been saved: {args.final_fig_path}")


if __name__ == "__main__":
    run_training()
