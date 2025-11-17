import torch
import numpy as np
from model_cn_hybrid import CNNLSTM_Hybrid  # ✅ Correct model class name

# ========== Device Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ========== Load Input Arrays ==========
X_seq = np.load("X_seq_unseen.npy")
X_feat = np.load("X_feat_unseen.npy")

X_seq = torch.tensor(X_seq, dtype=torch.float32).to(device)
X_feat = torch.tensor(X_feat, dtype=torch.float32).to(device)

# ========== Load Model ==========
model = CNNLSTM_Hybrid()
model.load_state_dict(torch.load("model_weights.pth", map_location=device))
model = model.to(device)
model.eval()

# ========== Run Prediction ==========
print("🚀 Starting prediction on unseen chickpea data...")
with torch.no_grad():
    preds = model(X_seq, X_feat)
    predicted_labels = torch.argmax(preds, dim=1).cpu().numpy()

# ========== Save Predictions ==========
import pandas as pd
pd.DataFrame(predicted_labels, columns=["Prediction"]).to_csv("output/predicted_unseen_labels.csv", index=False)
print("✅ Predictions saved to 'output/predicted_unseen_labels.csv'")

