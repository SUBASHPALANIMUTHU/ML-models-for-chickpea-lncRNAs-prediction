import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch
from collections import defaultdict
from subprocess import Popen, PIPE

# ========== Configuration =========
input_fasta = "data/chickpea_data.fa"
max_len = 300  # same as used during training
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# ========== Utility Functions ==========

def one_hot_encode(seq, max_len=300):
    mapping = {"A": 0, "C": 1, "G": 2, "U": 3}
    seq = seq.upper().replace("T", "U")
    encoded = np.zeros((max_len, 4), dtype=np.float32)
    for i, base in enumerate(seq[:max_len]):
        if base in mapping:
            encoded[i, mapping[base]] = 1.0
    return encoded

def gc_content(seq):
    seq = seq.upper().replace("T", "U")
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq) if len(seq) > 0 else 0

def orf_length(seq):
    seq = seq.upper().replace("T", "U")
    start_codon = "AUG"
    stop_codons = ["UAA", "UGA", "UAG"]
    max_len = 0
    for frame in range(3):
        length = 0
        in_orf = False
        for i in range(frame, len(seq) - 2, 3):
            codon = seq[i:i+3]
            if not in_orf and codon == start_codon:
                in_orf = True
                length = 3
            elif in_orf:
                length += 3
                if codon in stop_codons:
                    max_len = max(max_len, length)
                    in_orf = False
                    length = 0
        if in_orf:
            max_len = max(max_len, length)
    return max_len

def orf_coverage(seq):
    ol = orf_length(seq)
    return ol / len(seq) if len(seq) > 0 else 0

def fickett_score(seq):
    seq = seq.upper().replace("T", "U")
    counts = {'A':[0,0,0], 'C':[0,0,0], 'G':[0,0,0], 'U':[0,0,0]}
    for i, nt in enumerate(seq):
        if nt in counts:
            counts[nt][i % 3] += 1
    total = len(seq)
    if total == 0:
        return 0
    freq = {nt: sum(pos_counts)/total for nt, pos_counts in counts.items()}
    pos_bias = []
    for nt in counts:
        pos_counts = counts[nt]
        max_c = max(pos_counts)
        min_c = min(pos_counts) if min(pos_counts) > 0 else 1
        pos_bias.append(max_c / min_c)
    pos_bias_score = sum(pos_bias) / 4
    content_score = sum(freq.values())
    return pos_bias_score * content_score

def hexamer_score(seq, hexamer_scores):
    seq = seq.upper().replace("T", "U")
    scores = [hexamer_scores.get(seq[i:i+6], 0) for i in range(len(seq)-5)]
    return sum(scores)/len(scores) if scores else 0

def get_mfe(seq):
    seq = seq.upper().replace("T", "U")
    try:
        p = Popen(['RNAfold', '--noPS'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        output, _ = p.communicate(input=seq.encode())
        lines = output.decode().strip().split('\n')
        if len(lines) >= 2:
            mfe_str = lines[1].split('(')[-1].replace(')', '').strip()
            return float(mfe_str)
    except:
        pass
    return 0.0

def build_dummy_hexamer_table():
    return defaultdict(int)  # All hexamer scores = 0 (neutral), or load from training if available

# ========== Main Process ==========

print("🚀 Starting feature extraction for unseen chickpea data...")

hexamer_table = build_dummy_hexamer_table()

X_seq = []
X_feat = []
seq_ids = []

for record in SeqIO.parse(input_fasta, "fasta"):
    seq_id = record.id
    seq = str(record.seq)
    onehot = one_hot_encode(seq, max_len=max_len)
    gc = gc_content(seq)
    orf = orf_length(seq)
    cov = orf_coverage(seq)
    fick = fickett_score(seq)
    hexs = hexamer_score(seq, hexamer_table)
    mfe = get_mfe(seq)

    X_seq.append(onehot)
    X_feat.append([gc, orf / max_len, cov, fick, hexs, mfe])
    seq_ids.append(seq_id)

X_seq = np.array(X_seq)
X_feat = np.array(X_feat)

# ========== Save ==========
np.save("X_seq_unseen.npy", X_seq)
np.save("X_feat_unseen.npy", X_feat)
pd.DataFrame(X_feat, columns=["GC", "ORF_len", "ORF_cov", "Fickett", "Hexamer", "MFE"], index=seq_ids)\
  .to_csv(os.path.join(output_dir, "features_unseen.csv"))

print("✅ Feature files saved:")
print(" - X_seq_unseen.npy")
print(" - X_feat_unseen.npy")
print(" - output/features_unseen.csv")

