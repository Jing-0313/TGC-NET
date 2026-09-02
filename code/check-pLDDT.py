import os
import glob
import pandas as pd
from Bio import PDB
import numpy as np
from tqdm import tqdm


pdb_folder = "/home/ys/new/xvjing/data/topt1/test-pdb"
output_csv = "/home/ys/new/xvjing/data/topt1/291-quality.csv"


def get_avg_plddt(pdb_path):
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_path)
        plddt_scores = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == "CA":
                            plddt_scores.append(atom.get_bfactor())
        if not plddt_scores:
            return 0.0
        return np.mean(plddt_scores)
    except Exception as e:
        print(f"Error reading {pdb_path}: {e}")
        return 0.0


def main():
    pdb_files = glob.glob(os.path.join(pdb_folder, "*.pdb"))
    print(f"{len(pdb_files)} PDB files ..")
    results = []
    for file_path in tqdm(pdb_files):
        file_name = os.path.basename(file_path)
        avg_score = get_avg_plddt(file_path)
        if avg_score > 0.9:
            rating = "Excellent"
        elif avg_score > 0.7:
            rating = "Good"
        elif avg_score > 0.5:
            rating = "Fair"
        else:
            rating = "Poor"
        results.append({
            "Filename": file_name,
            "Avg_pLDDT": round(avg_score, 2),
            "Rating": rating
        })
    df = pd.DataFrame(results)
    df = df.sort_values(by="Avg_pLDDT", ascending=False)
    df.to_csv(output_csv, index=False)
    print(df["Rating"].value_counts())

if __name__ == "__main__":
    main()