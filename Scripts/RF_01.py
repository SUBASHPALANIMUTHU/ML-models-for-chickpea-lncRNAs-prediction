import os
import pandas as pd
import numpy as np
from Bio import SeqIO
import subprocess
import tempfile

def mfe(seq):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
        tmp.write(seq + '\n')
        tmp_path = tmp.name
    try:
        result = subprocess.run(["RNAfold", tmp_path], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            energy_line = lines[1]
            energy = float(energy_line.split('(')[-1].replace(')', ''))
            return energy
    except Exception as e:
        print("ViennaRNA mfe error:", e)
    return 0.0

def build_hexamer_table(fasta_file):
    hexamer_counts = {}
    total = 0
    for rec in SeqIO.parse(fasta_file, "fasta"):
        seq = str(rec.seq).upper()
        for i in range(len(seq) - 5):
            hexamer = seq[i:i+6]
            if all(c in "ACGT" for c in hexamer):
                hexamer_counts[hexamer] = hexamer_counts.get(hexamer, 0) + 1
                total += 1
    for h in hexamer_counts:
        hexamer_counts[h] /= total
    return hexamer_counts

def compute_hexamer_score(seq, hexamer_table):
    seq = seq.upper()
    scores = []
    for i in range(len(seq) - 5):
        hexamer = seq[i:i+6]
        if hexamer in hexamer_table:
            scores.append(hexamer_table[hexamer])
    return np.mean(scores) if scores else 0.0

def gc_content(seq):
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / len(seq)

def orf_length(seq):
    seq = seq.upper()
    orfs = []
    start_codon = "ATG"
    for i in range(len(seq) - 3):
        if seq[i:i+3] == start_codon:
            orfs.append(len(seq) - i)
    return max(orfs) if orfs else 0

def fickett_score(seq):
    at = (seq.count('A') + seq.count('T')) / len(seq)
    gc = (seq.count('G') + seq.count('C')) / len(seq)
    return 0.5 * at + 0.5 * gc

def sequence_length(seq):
    return len(seq)

def extract_features(seq, hexamer_table):
    seq = seq.upper()
    return {
        "GC_Content": gc_content(seq),
        "ORF_Length": orf_length(seq),
        "Fickett_Score": fickett_score(seq),
        "Hexamer_Score": compute_hexamer_score(seq, hexamer_table),
        "MFE": mfe(seq),
        "Sequence_Length": sequence_length(seq),
    }

def load_fasta_features(filepath, label, cache_csv, hexamer_table):
    if os.path.exists(cache_csv):
        print(f"✅ Skipping feature extraction: {cache_csv}")
        return pd.read_csv(cache_csv)

    print(f"🔍 Extracting features from {filepath}...")
    records = list(SeqIO.parse(filepath, "fasta"))
    data = []
    for i, rec in enumerate(records):
        if i % 1000 == 0:
            print(f"  ...{i}/{len(records)} sequences")
        feats = extract_features(str(rec.seq), hexamer_table)
        feats["Label"] = label
        feats["ID"] = rec.id
        data.append(feats)

    df = pd.DataFrame(data)
    df.to_csv(cache_csv, index=False)
    print(f"✅ Features saved to {cache_csv}")
    return df

if __name__ == "__main__":
    data_dir = "data"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    hexamer_table = build_hexamer_table(f"{data_dir}/coding_data.fa")

    load_fasta_features(f"{data_dir}/lnc_RNA_data.fa", 1, f"{output_dir}/lnc_features.csv", hexamer_table)
    load_fasta_features(f"{data_dir}/coding_data.fa", 0, f"{output_dir}/coding_features.csv", hexamer_table)
    load_fasta_features(f"{data_dir}/chickpea-data.fa", -1, f"{output_dir}/chickpea_unseen_features.csv", hexamer_table)
