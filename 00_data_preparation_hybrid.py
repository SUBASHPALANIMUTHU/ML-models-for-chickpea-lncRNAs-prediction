import os
import torch
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from collections import defaultdict

# === One-hot encode RNA sequences ===
def one_hot_encode(seq, max_len=300):
    mapping = {"A": 0, "C": 1, "G": 2, "U": 3}
    encoded = np.zeros((max_len, 4), dtype=np.float32)
    seq = seq.upper().replace("T", "U")
    for i, base in enumerate(seq[:max_len]):
        if base in mapping:
            encoded[i, mapping[base]] = 1.0
    return encoded

# === Fickett score calculation ===
def fickett_score(seq):
    seq = seq.upper().replace("T", "U")
    # Frequencies for A,C,G,U at positions mod 3 = 0,1,2
    counts = {'A':[0,0,0], 'C':[0,0,0], 'G':[0,0,0], 'U':[0,0,0]}
    for i, nt in enumerate(seq):
        if nt in counts:
            counts[nt][i%3] += 1
    # Calculate position bias and content bias
    total = len(seq)
    if total == 0:
        return 0
    freq = {nt: sum(pos_counts)/total for nt, pos_counts in counts.items()}
    # position bias: max pos freq / min pos freq for each nt
    pos_bias = []
    for nt in counts:
        pos_counts = counts[nt]
        max_c = max(pos_counts)
        min_c = min(pos_counts) if min(pos_counts) > 0 else 1
        pos_bias.append(max_c / min_c)
    pos_bias_score = sum(pos_bias)/4
    content_score = sum(freq.values())
    score = pos_bias_score * content_score
    return score

# === Hexamer score calculation ===

def build_hexamer_table(coding_seqs, noncoding_seqs):
    coding_counts = defaultdict(int)
    noncoding_counts = defaultdict(int)

    # Count hexamers in coding seqs
    for _, seq in coding_seqs:
        seq = seq.upper().replace("T", "U")
        for i in range(len(seq) - 5):
            hexamer = seq[i:i+6]
            coding_counts[hexamer] += 1

    # Count hexamers in noncoding seqs
    for _, seq in noncoding_seqs:
        seq = seq.upper().replace("T", "U")
        for i in range(len(seq) - 5):
            hexamer = seq[i:i+6]
            noncoding_counts[hexamer] += 1

    # Calculate hexamer score table: log2((coding + 1)/(noncoding + 1))
    hexamer_scores = {}
    all_hexamers = set(list(coding_counts.keys()) + list(noncoding_counts.keys()))
    for hexamer in all_hexamers:
        c = coding_counts[hexamer]
        nc = noncoding_counts[hexamer]
        hexamer_scores[hexamer] = np.log2((c + 1) / (nc + 1))
    return hexamer_scores

def hexamer_score(seq, hexamer_scores):
    seq = seq.upper().replace("T", "U")
    scores = []
    for i in range(len(seq) - 5):
        hexamer = seq[i:i+6]
        scores.append(hexamer_scores.get(hexamer, 0))
    if scores:
        return sum(scores) / len(scores)
    else:
        return 0

# === Other feature calculations ===

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
                    if length > max_len:
                        max_len = length
                    in_orf = False
                    length = 0
        if in_orf and length > max_len:
            max_len = length
    return max_len

def orf_coverage(seq):
    orf_len = orf_length(seq)
    return orf_len / len(seq) if len(seq) > 0 else 0

# === Load fasta sequences ===
def load_fasta(file_path):
    return [(record.id, str(record.seq)) for record in SeqIO.parse(file_path, "fasta")]

print("Loading sequences...")
lnc_seqs = load_fasta("data/lnc_RNA_data.fa")
coding_seqs = load_fasta("data/coding_data.fa")

print("Building hexamer table...")
hexamer_table = build_hexamer_table(coding_seqs, lnc_seqs)

print("Encoding sequences and extracting features...")

x_seq = []
x_feat = []
y = []

for i, (_, seq) in enumerate(lnc_seqs):
    x_seq.append(one_hot_encode(seq))
    feats = [
        gc_content(seq),
        orf_length(seq) / 300,       # normalized
        orf_coverage(seq),
        fickett_score(seq),
        hexamer_score(seq, hexamer_table)
    ]
    x_feat.append(feats)
    y.append(1)
    if i < 5:
        print(f"lncRNA example {i+1} features: {feats}")

for i, (_, seq) in enumerate(coding_seqs):
    x_seq.append(one_hot_encode(seq))
    feats = [
        gc_content(seq),
        orf_length(seq) / 300,
        orf_coverage(seq),
        fickett_score(seq),
        hexamer_score(seq, hexamer_table)
    ]
    x_feat.append(feats)
    y.append(0)
    if i < 5:
        print(f"mRNA example {i+1} features: {feats}")

x_seq = torch.tensor(x_seq)
x_feat = torch.tensor(x_feat, dtype=torch.float32)
y = torch.tensor(y)

print(f"Total samples: {len(y)}")

# Split 80/20 with stratify for balanced classes
from sklearn.model_selection import train_test_split
seq_train, seq_val, feat_train, feat_val, y_train, y_val = train_test_split(
    x_seq, x_feat, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")

os.makedirs("output", exist_ok=True)
torch.save((seq_train, feat_train, y_train), "output/train_dataset_hybrid.pt")
torch.save((seq_val, feat_val, y_val), "output/val_dataset_hybrid.pt")

print("✅ Hybrid data preparation complete and saved.")

