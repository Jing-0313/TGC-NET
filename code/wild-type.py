import os
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
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

    source_map = {}
    for k, v in state_dict.items():
        if "lora_" in k or "lora_A" in k or "lora_B" in k: continue
        clean_k = k.replace("base_model.model.", "").replace("module.", "").replace(".base_layer", "")
        source_map[clean_k] = v

    for target_k in model_keys:
        if target_k in source_map and model.state_dict()[target_k].shape == source_map[target_k].shape:
            new_state_dict[target_k] = source_map[target_k]

    model.load_state_dict(new_state_dict, strict=False)
    print(f" [{model_name}] weight loading completed.")


class MultimodalToptPredictor:
    def __init__(self, model_path, esm_name, structure_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.structure_dict = torch.load(structure_path, map_location='cpu',weights_only=False)

        self.tokenizer = AutoTokenizer.from_pretrained(esm_name)
        self.plm_model = AutoModel.from_pretrained(esm_name).to(self.device)
        self.model = MultimodalEnzymeModel(dim=self.plm_model.config.hidden_size).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device,weights_only=False)
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
        min_v, max_v = 0.0, 120.0

        print(f"Start predicting {len(ids)} Article data...")
        with torch.no_grad():
            for i in tqdm(range(0, len(ids), batch_size), desc="Predicting"):
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

        df['pred_topt'] = all_preds
        return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/home/ys/new/xvjing/case-study/case-test.csv')
    parser.add_argument('--output', default='/home/ys/new/xvjing/case-study/0.614result-topt.csv')
    parser.add_argument('--model_path', default='/home/ys/new/xvjing/path/topt5/6topt_best_model_0.614.pth')
    parser.add_argument('--structure_path', default='/home/ys/new/xvjing/case-study/structures.pt')
    parser.add_argument('--esm_name', default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    predictor = MultimodalToptPredictor(args.model_path, args.esm_name, args.structure_path)
    df = pd.read_csv(args.input)
    df['uniprot_id'] = df['uniprot_id'].astype(str)
    df = df[df['uniprot_id'].isin(predictor.structure_dict.keys())].reset_index(drop=True)

    res_df = predictor.predict(df, batch_size=args.batch_size)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    res_df.to_csv(args.output, index=False)
    print(f"The predicted results have been successfully saved to: {args.output}")


if __name__ == "__main__":
    main()