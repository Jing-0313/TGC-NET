import os
import numpy as np
import sys
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import math
from math import sqrt
import requests
import subprocess
from collections import Counter
from Bio import PDB


def parse_pdb_coords(pdb_path, max_len=None):
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]  # 取第一个模型

        coords = []
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                try:
                    ca_atom = residue['CA']
                    coords.append(ca_atom.get_coord())
                except KeyError:
                    coords.append([0.0, 0.0, 0.0])
        coords_tensor = torch.tensor(np.array(coords, dtype=np.float32))
        if max_len is not None:
            if len(coords_tensor) > max_len:
                coords_tensor = coords_tensor[:max_len]
        return coords_tensor
    except Exception as e:
        print(f"Failed to parse PDB: {pdb_path} - {e}")
        return None


def get_rmse(x, y):
    x, y = np.array(x), np.array(y)
    mse = np.mean((x - y) ** 2)
    return round(sqrt(mse), 6)

def get_r2(x, y):
    """计算 R2 分数"""
    x, y = np.array(x), np.array(y)
    return round(r2_score(x, y), 6)

def get_mae(x, y):
    """计算平均绝对误差"""
    x, y = np.array(x), np.array(y)
    return round(mean_absolute_error(x, y), 6)

def get_seq(ID):
    url = "https://www.uniprot.org/uniprot/%s.fasta" % ID
    try:
        data = requests.get(url, timeout=10)
        if data.status_code != 200:
            return 'NaN'
        lines = data.text.strip().split("\n")
        if len(lines) > 1:
            return "".join(lines[1:])
        return 'NaN'
    except:
        return 'NaN'


def split_table(table, ratio, seed=42):
    idx = list(table.index)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(idx)
    num_split = int(len(idx) * ratio)
    idx_test, idx_train = idx[:num_split], idx[num_split:]
    train_table = table.loc[idx_train].reset_index(drop=True)
    test_table = table.loc[idx_test].reset_index(drop=True)
    return train_table, test_table


def rescale_targets(target_values, x_max, x_min):
    if x_max == x_min:
        return [0.0] * len(target_values)
    return [(x - x_min) / (x_max - x_min) for x in target_values]


def run(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def dump_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def get_aac(seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    length = len(seq)
    output = {}
    if length == 0: return output
    for aa in AA:
        output['AAC_' + aa] = seq.count(aa) / length
    return output


def get_dpc(seq):
    output = {}
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    length = len(seq)
    if length <= 1: return output
    DPs = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    for dp in DPs:
        output['DPC_' + dp] = seq.count(dp) / (length - 1)
    return output


def get_aacdpc(seq):
    if 'X' in seq:
        counts = Counter(seq)
        most_common = counts.most_common()
        most_aa = 'A'
        for aa, _ in most_common:
            if aa != 'X':
                most_aa = aa
                break
        seq = seq.replace('X', most_aa)

    results = get_aac(seq)
    results.update(get_dpc(seq))
    return results