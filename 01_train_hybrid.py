import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_cn_hybrid import CNNLSTM_Hybrid

# Load training data: sequences, features, labels
X_seq, X_feat, y = torch.load("output/train_dataset_hybrid.pt")

# Dataset and DataLoader
train_dataset = TensorDataset(X_seq, X_feat, y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model, Loss, Optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNNLSTM_Hybrid().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 30

for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for xb_seq, xb_feat, yb in train_loader:
        xb_seq, xb_feat, yb = xb_seq.to(device), xb_feat.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(xb_seq, xb_feat)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"✅ Epoch {epoch}/{EPOCHS} Loss: {total_loss/len(train_loader):.4f}")

# Save the model weights
torch.save(model.state_dict(), "models/lncRNA_model_hybrid.pt")
print("\n✅ Model saved to models/lncRNA_model_hybrid.pt")

