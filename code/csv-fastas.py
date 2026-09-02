import pandas as pd
import os


def merge_csv_to_fastas(csv_files, output_folder="all_proteins"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"The directory has been created: {output_folder}")
    else:
        print(f"File '{output_folder}' already exists, new files will be stored in it.")
    total_count = 0
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Skip: File not found {csv_file}")
            continue
        print(f"Processing file: {csv_file} ...")
        df = pd.read_csv(csv_file)
        if 'uniprot_id' not in df.columns or 'sequence' not in df.columns:
            print(f"Error：File {csv_file} Lacks 'uniprot_id' or 'sequence'.")
            continue
        for _, row in df.iterrows():
            prot_id = str(row['uniprot_id'])
            sequence = str(row['sequence'])
            safe_id = "".join(x for x in prot_id if x.isalnum() or x in "._-")
            file_path = os.path.join(output_folder, f"{safe_id}.fasta")
            with open(file_path, "w") as f:
                f.write(f">{prot_id}\n")
                f.write(f"{sequence}\n")
            total_count += 1
            if total_count % 5000 == 0:
                print(f"Generated {total_count} files...")

    print(f"All files are saved in: {os.path.abspath(output_folder)}")


if __name__ == "__main__":
    files_to_process = ["/home/ys/new/xvjing/wendu/diffthermo_data.csv"]
    target_dir = "/home/ys/new/xvjing/wendu/fastas"
    merge_csv_to_fastas(files_to_process, target_dir)