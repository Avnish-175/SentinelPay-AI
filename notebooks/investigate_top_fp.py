from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "baseline"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "xgboost_no_ratio.json"
PREPROCESSOR_CONFIG_PATH = MODEL_DIR / "xgboost_no_ratio_preprocessing.json"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"

TARGET = "isFraud"
CATEGORICAL_FEATURES = ["type"]
NUMERICAL_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "transaction_day",
    "transaction_hour",
    "hour_sin",
    "hour_cos",
    "origin_is_zero",
    "destination_is_zero",
]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

DTYPES = {"type": "category", TARGET: "int8"}
DTYPES.update({name: "float32" for name in NUMERICAL_FEATURES})

def main():
    print("Loading preprocessor and model...")
    with open(PREPROCESSOR_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    categories = [config["categorical_categories"]]
    encoder = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    categories=categories,
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", "passthrough", NUMERICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    print("Loading validation data...")
    val_cols = FEATURES + [TARGET, "step"]
    # Check if raw ratio features or account IDs are present in validation.csv
    sample_df = pd.read_csv(VALIDATION_PATH, nrows=5)
    print("Validation columns:", list(sample_df.columns))

    val_df = pd.read_csv(VALIDATION_PATH)
    print(f"Validation shape: {val_df.shape}")

    encoder.fit(val_df[FEATURES])
    feature_names = encoder.get_feature_names_out()
    cleaned_feature_names = [
        name.replace("categorical__", "").replace("numerical__", "")
        for name in feature_names
    ]

    X_sparse = encoder.transform(val_df[FEATURES]).tocsr().astype(np.float32)
    y = val_df[TARGET].to_numpy(dtype=np.int8)

    probs = model.predict_proba(X_sparse)[:, 1]

    # Find highest scored legitimate transaction (y == 0)
    legit_mask = (y == 0)
    legit_indices = np.flatnonzero(legit_mask)
    top_fp_idx = legit_indices[np.argmax(probs[legit_indices])]
    top_fp_prob = float(probs[top_fp_idx])

    print(f"\nTop FP Validation Index: {top_fp_idx}")
    print(f"Top FP Predicted Probability: {top_fp_prob:.8f}")

    raw_row = val_df.iloc[top_fp_idx].to_dict()
    print("\nRaw Row Data:")
    for k, v in raw_row.items():
        print(f"  {k}: {v}")

    # Compute SHAP values for this specific row
    explainer = shap.TreeExplainer(model)
    fp_sparse = X_sparse[top_fp_idx : top_fp_idx + 1]
    shap_explanation = explainer(fp_sparse)

    base_value = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])
    shap_values = shap_explanation.values[0]

    contributions = []
    for f_name, raw_val, shap_val in zip(cleaned_feature_names, fp_sparse.toarray()[0], shap_values):
        contributions.append({
            "feature": f_name,
            "transformed_value": float(raw_val),
            "shap_value": float(shap_val)
        })

    contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    print(f"\nExplainer Base Value (margin/logit): {base_value:.6f}")
    print(f"Sum of base_value + SHAP: {base_value + np.sum(shap_values):.6f}")
    margin = base_value + np.sum(shap_values)
    prob_from_margin = 1 / (1 + np.exp(-margin))
    print(f"Logistic sigmoid(margin): {prob_from_margin:.8f}")

    print("\nIndividual SHAP Contributions:")
    for c in contributions:
        print(f"  {c['feature']:<25} | Transformed: {c['transformed_value']:<12.4f} | SHAP: {c['shap_value']:+.6f}")

    output_result = {
        "validation_row_index": int(top_fp_idx),
        "predicted_probability": top_fp_prob,
        "base_value_logit": base_value,
        "sum_shap_margin": float(margin),
        "sigmoid_margin_prob": float(prob_from_margin),
        "raw_record": {k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, int)) else str(v))) for k, v in raw_row.items()},
        "shap_contributions": contributions,
    }

    out_file = REPORT_DIR / "top_false_positive_investigation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_result, f, indent=4)
    print(f"\nSaved analysis to: {out_file}")

if __name__ == "__main__":
    main()
