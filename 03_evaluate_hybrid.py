from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_cn_hybrid import CNNLSTM_Hybrid

# Load validation data
X_seq, X_feat, y = torch.load("output/val_dataset_hybrid.pt")

val_dataset = TensorDataset(X_seq, X_feat, y)
val_loader = DataLoader(val_dataset, batch_size=64)

# Load model
model = CNNLSTM_Hybrid()
model.load_state_dict(torch.load("models/lncRNA_model_hybrid.pt"))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

preds, trues = [], []

with torch.no_grad():
    for xb_seq, xb_feat, yb in val_loader:
        xb_seq, xb_feat = xb_seq.to(device), xb_feat.to(device)
        out = model(xb_seq, xb_feat)
        preds.extend(torch.argmax(out, dim=1).cpu().numpy())
        trues.extend(yb.numpy())

print("Classification Report:\n", classification_report(trues, preds, target_names=["mRNA", "lncRNA"]))

