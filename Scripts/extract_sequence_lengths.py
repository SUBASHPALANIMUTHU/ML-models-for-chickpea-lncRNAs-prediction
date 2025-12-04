from Bio import SeqIO
import pandas as pd
import os

# Set input and output
input_fasta = "data/chickpea_data.fa"
output_csv = "output/chickpea_sequence_lengths.csv"

# Make sure output folder exists
os.makedirs("output", exist_ok=True)

# Extract ID and sequence length
records = []
for record in SeqIO.parse(input_fasta, "fasta"):
    records.append({"ID": record.id, "Sequence_Length": len(record.seq)})

# Save to CSV
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)

print(f"✅ Saved: {output_csv}")
