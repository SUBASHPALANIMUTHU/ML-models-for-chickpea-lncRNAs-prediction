import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import shap
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_prob, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_true, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(outpath)
    plt.close()

def plot_shap_summary(model, X, outpath):
    print("🔍 Computing SHAP values (this might take a while)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # shap_values can be list of arrays for multi-class; for binary use shap_values[1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap.summary_plot(shap_values[1], X, show=False)
    else:
        shap.summary_plot(shap_values, X, show=False)

    plt.savefig(outpath)
    plt.close()
    print(f"📊 SHAP summary plot saved: {outpath}")

if __name__ == "__main__":
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "rf_lnc_model.joblib")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    shap_path = os.path.join(output_dir, "shap_summary_plot.png")

    # Load features CSV
    print("🔍 Loading feature data...")
    lnc_df = pd.read_csv(f"{output_dir}/lnc_features.csv")
    cds_df = pd.read_csv(f"{output_dir}/coding_features.csv")

    # Combine and shuffle
    full_df = pd.concat([lnc_df, cds_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Prepare feature matrix X and label vector y
    X = full_df.drop(columns=["Label", "ID"])
    y = full_df["Label"]

    # 80-20 train-test split
    print("🚀 Splitting data: 80% train, 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Compute sample weights for balanced training
    weights = compute_sample_weight("balanced", y_train)

    # Initialize Random Forest with 30 trees
    print("🚀 Training Random Forest with 30 estimators (like 30 epochs)...")
    rf = RandomForestClassifier(
        n_estimators=30,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train, sample_weight=weights)

    # Evaluate on test set
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("\n📊 Evaluation on 20% test set:")
    print(classification_report(y_test, y_pred))
    print(f"🎯 ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # Save model
    joblib.dump(rf, model_path)
    print(f"✅ Model saved to {model_path}")

    # Plot ROC curve
    plot_roc_curve(y_test, y_prob, roc_path)
    print(f"📈 ROC curve saved: {roc_path}")

    # Plot SHAP summary (using train data for SHAP to avoid shape mismatch)
    plot_shap_summary(rf, X_train, shap_path)
