import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# Step 1: Load feature data
# --------------------
df_lnc = pd.read_csv("lnc_features.csv")
df_coding = pd.read_csv("coding_features.csv")
df_unseen = pd.read_csv("chickpea_unseen_featuress.csv")

# Assign labels
df_lnc["Label"] = 1
df_coding["Label"] = 0

# Merge and shuffle
df_all = pd.concat([df_lnc, df_coding], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

# --------------------
# Step 2: Feature and label extraction
# --------------------
feature_cols = ["GC_Content", "ORF_Length", "Fickett_Score", "Hexamer_Score", "MFE", "Sequence_Length"]
X = df_all[feature_cols]
y = df_all["Label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------
# Step 3: Train model
# --------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --------------------
# Step 4: Evaluate model
# --------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n=== Evaluation Metrics ===")
print(f"Accuracy       : {acc:.4f}")
print(f"Precision      : {prec:.4f}")
print(f"Recall         : {rec:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"ROC AUC        : {auc:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------
# Step 5: Plot ROC Curve
# --------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("output/roc_curve.png")
plt.close()

# --------------------
# Step 6: Predict on unseen data
# --------------------
X_unseen = df_unseen[feature_cols]
pred_unseen = model.predict(X_unseen)
proba_unseen = model.predict_proba(X_unseen)[:, 1]

df_unseen["Predicted_Label"] = pred_unseen
df_unseen["lncRNA_Probability"] = proba_unseen
df_unseen.to_csv("output/chickpea_unseen_predictions.csv", index=False)

print("✅ Model trained and evaluated.")
print("📝 Unseen chickpea prediction saved to: output/chickpea_unseen_predictions.csv")
print("📊 ROC curve saved to: output/roc_curve.png")
