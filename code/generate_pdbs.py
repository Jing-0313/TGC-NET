import os
import torch
import argparse
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding
from Bio import SeqIO  # 使用 BioPython 解析 FASTA 更稳健
import sys
import re

TRUNK_LEN = 64

def check_cuda_health():
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return "cpu"
    try:
        t = torch.tensor([1.0]).cuda()
        _ = t + 1
        return "cuda"
    except RuntimeError as e:
        print("Graphics card status abnormal, please restart the terminal or machine!")
        sys.exit(1)


def setup_model(model_name="facebook/esmfold_v1"):
    device = check_cuda_health()
    print(f"Loading ESMFold model: {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForProteinFolding.from_pretrained(model_name, low_cpu_mem_usage=True)
    model.trunk_sequence_length = TRUNK_LEN
    model = model.eval().to(device)
    return tokenizer, model, device


def sanitize_sequence(seq):
    if not seq: return ""
    seq = str(seq).upper().strip()
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "A", seq)
    return seq


def process_fasta_folder(input_dir, output_dir, tokenizer, model, device):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fasta_files = glob.glob(os.path.join(input_dir, "*.fasta"))
    print(f"Scan directory: {input_dir}")
    print(f"Find {len(fasta_files)} files")
    success_count = 0
    skip_count = 0
    for file_path in tqdm(fasta_files, desc="ESMFold Prediction", mininterval=10):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(output_dir, f"{file_name}.pdb")
        if os.path.exists(save_path):
            skip_count += 1
            continue
        try:
            with open(file_path, "r") as handle:
                record = next(SeqIO.parse(handle, "fasta"))
            sequence = sanitize_sequence(str(record.seq))
            if len(sequence) > 2000:
                print(f"Skip ultra long sequences {file_name}: length {len(sequence)}")
                continue
            inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            pdb_structure = model.output_to_pdb(outputs)[0]
            with open(save_path, "w") as f:
                f.write(pdb_structure)
            success_count += 1
            del inputs, outputs, pdb_structure
            torch.cuda.empty_cache()
        except StopIteration:
            print(f"Empty file: {file_name}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM Skip: {file_name} (length {len(sequence)})")
                torch.cuda.empty_cache()
            else:
                print(f"Runtime error {file_name}: {e}")
        except Exception as e:
            print(f"Unknown error {file_name}: {e}")
    print(f"Processing completed!")
    print(f"   - Newly generated: {success_count}")
    print(f"   - Skip existing ones: {skip_count}")
    print(f"   - The results are saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/home/ys/new/xvjing/wendu/fastas')
    parser.add_argument('--output_dir', default='/home/ys/new/xvjing/wendu/pdbs')
    args = parser.parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    tokenizer, model, device = setup_model()
    process_fasta_folder(args.input_dir, args.output_dir, tokenizer, model, device)