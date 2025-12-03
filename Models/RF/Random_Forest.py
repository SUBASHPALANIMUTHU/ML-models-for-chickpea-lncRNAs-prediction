"""
Random Forest Model for lncRNA Prediction
-----------------------------------------


Author: SUBASH P
MSc Research Project – CCS Haryana Agricultural University
Date: 2025

Steps:
1. Load and label feature datasets (lncRNA vs mRNA).
2. Preprocess features (cleaning, scaling).
3. Train a Random Forest classifier.
4. Evaluate using standard ML metrics.
5. Save trained model, scaler, and feature importance table.
6. Perform predictions on unseen chickpea transcripts.
7. Optional SHAP explainability (safe subset to avoid overload).

"""

# =============================================================================
# Imports
# =============================================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Step 1 — Load Feature Datasets
# =============================================================================

lnc_file = "lnc_features.csv"
coding_file = "coding_features.csv"
unseen_file = "chickpea_unseen_featuress.csv"

print("📥 Loading feature files...")
lnc_df = pd.read_csv(lnc_file)
coding_df = pd.read_csv(coding_file)
unseen_df = pd.read_csv(unseen_file)

# Add labels: 1 = lncRNA, 0 = mRNA
lnc_df["Label"] = 1
coding_df["Label"] = 0

# Merge both datasets
full_df = pd.concat([lnc_df, coding_df], ignore_index=True)
print(f"✓ Combined dataset shape: {full_df.shape}")

# =============================================================================
# Step 2 — Preprocessing
# =============================================================================

# Drop non-feature columns (ID or metadata)
X = full_df.drop(columns=["Label", "ID"], errors="ignore")
y = full_df["Label"]

# Clean invalid values
before_rows = X.shape[0]
X = X.replace([np.inf, -np.inf], np.nan).dropna()
after_rows = X.shape[0]

print(f"✓ Removed {before_rows - after_rows} rows containing NaN/Inf values.")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for reproducibility
joblib.dump(scaler, "rf_scaler.pkl")
print("✓ Scaler saved as rf_scaler.pkl")

# =============================================================================
# Step 3 — Train/Test Split
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, stratify=y, random_state=42
)
print("✓ Train/test split completed")

# =============================================================================
# Step 4 — Random Forest Training
# =============================================================================

print("\n🚀 Training Random Forest model...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("✓ Model training complete")

# Save model
joblib.dump(rf_model, "rf_model.pkl")
print("✓ Model saved as rf_model.pkl")

# =============================================================================
# Step 5 — Evaluation
# =============================================================================

print("\n📊 Model Evaluation")

y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))
print("ROC AUC:", round(roc_auc_score(y_test, y_proba), 4))

# Confusion Matrix Plot
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(conf_mat, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["mRNA", "lncRNA"])
plt.yticks([0, 1], ["mRNA", "lncRNA"])
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png", dpi=300)
plt.close()
print("✓ Confusion matrix saved as rf_confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig("rf_roc_curve.png", dpi=300)
plt.close()
print("✓ ROC curve saved as rf_roc_curve.png")

# =============================================================================
# Step 6 — Feature Importance
# =============================================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance.to_csv("rf_feature_importance.csv", index=False)
print("✓ Feature importance saved as rf_feature_importance.csv")

# =============================================================================
# Step 7 — Predict on Unseen Chickpea Data
# =============================================================================

print("\n🔍 Predicting on unseen chickpea transcripts...")

unseen_ids = unseen_df["ID"] if "ID" in unseen_df else range(len(unseen_df))
X_unseen = unseen_df.drop(columns=["ID"], errors="ignore")
X_unseen_scaled = scaler.transform(X_unseen)

unseen_preds = rf_model.predict(X_unseen_scaled)

unseen_out = pd.DataFrame({
    "ID": unseen_ids,
    "Predicted_Label": unseen_preds
})
unseen_out.to_csv("chickpea_lncRNA_predictions.csv", index=False)
print("✓ Predictions saved as chickpea_lncRNA_predictions.csv")

# =============================================================================
# Step 8 — SHAP Explainability (optional but important for publication)
# =============================================================================

print("\n📘 Running SHAP explainability (sampling 300 instances)...")

try:
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_train[:300])

    shap.summary_plot(
        shap_values[1],
        X_train[:300],
        feature_names=X.columns,
        show=False
    )
    plt.tight_layout()
    plt.savefig("rf_shap_summary.png", dpi=300)
    plt.close()
    print("✓ SHAP summary saved as rf_shap_summary.png")

except Exception as e:
    print("⚠️ SHAP skipped due to error:", e)

print("\n🎉 All tasks completed successfully!")
