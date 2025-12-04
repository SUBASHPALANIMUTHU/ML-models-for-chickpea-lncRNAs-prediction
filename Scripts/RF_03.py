import os
import pandas as pd
import joblib

if __name__ == "__main__":
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    model_path = f"{output_dir}/rf_lnc_model.joblib"
    unseen_features_path = f"{output_dir}/chickpea_unseen_features.csv"
    prediction_output_path = f"{output_dir}/chickpea_unseen_predictions.csv"

    if os.path.exists(prediction_output_path):
        print(f"✅ Predictions already exist, skipping: {prediction_output_path}")
        exit()

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Please run RF_02.py first.")

    if not os.path.exists(unseen_features_path):
        raise FileNotFoundError("Unseen features not found. Please run RF_01.py first.")

    print("🔮 Loading model and unseen data features...")
    model = joblib.load(model_path)
    unseen_df = pd.read_csv(unseen_features_path)

    X_unseen = unseen_df.drop(columns=["Label", "ID"])
    predictions = model.predict(X_unseen)
    probabilities = model.predict_proba(X_unseen)[:, 1]

    unseen_df["Predicted_Label"] = predictions
    unseen_df["Predicted_Probability"] = probabilities

    unseen_df.to_csv(prediction_output_path, index=False)
    print(f"✅ Predictions saved to {prediction_output_path}")
