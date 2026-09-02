import os
import torch
import argparse
from tqdm import tqdm
import warnings
import glob
from functions import parse_pdb_coords

warnings.filterwarnings("ignore")


def process_esm_directory(pdb_dir, output_path):
    if not os.path.exists(pdb_dir):
        raise FileNotFoundError(f"Cannot find PDB directory: {pdb_dir}")

    structure_dict = {}

    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))

    print(f"Scanning PDB output directory: {pdb_dir}")
    print(f"Find {len(pdb_files)} pt files...")

    valid_count = 0

    for full_path in tqdm(pdb_files, desc="Packing to .pt"):
        filename = os.path.basename(full_path)
        pdb_id = os.path.splitext(filename)[0]
        coords = parse_pdb_coords(full_path)
        if coords is not None:
            structure_dict[pdb_id] = coords
            valid_count += 1
        else:
            print(f"Analysis failed or empty coordinates: {pdb_id}")

    print(f"Save {valid_count} effective structure to {output_path} ...")
    torch.save(structure_dict, output_path)
    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', default='/home/ys/new/xvjing/wendu/pdbs')
    parser.add_argument('--save_path', default='/home/ys/new/xvjing/wendu/diffthermo.pt')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    process_esm_directory(args.pdb_dir, args.save_path)