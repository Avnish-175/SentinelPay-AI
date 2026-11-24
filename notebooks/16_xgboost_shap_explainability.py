from pathlib import Path
import gc
import json

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# SHAP EXPLAINABILITY FOR THE SELECTED NO-RATIO XGBOOST MODEL
# Uses validation data for explanations; does not retrain or
# evaluate/tune the model on the chronological test split.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "baseline"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures" / "shap"

MODEL_PATH = MODEL_DIR / "xgboost_no_ratio.json"
PREPROCESSOR_CONFIG_PATH = MODEL_DIR / "xgboost_no_ratio_preprocessing.json"
METRICS_PATH = REPORT_DIR / "xgboost_no_ratio_metrics.json"
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

# A reproducible, manageable validation sample for global explanations.
GLOBAL_SAMPLE_SIZE = 2_000
RANDOM_STATE = 42

DTYPES = {"type": "category", TARGET: "int8"}
DTYPES.update({name: "float32" for name in NUMERICAL_FEATURES})


def to_float32_csr(matrix):
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))


def main():
    print("\n========== SENTINELPAY XGBOOST SHAP ANALYSIS ==========")
    print("XGBoost:", xgb.__version__)
    print("SHAP:", shap.__version__)

    for path in (MODEL_PATH, PREPROCESSOR_CONFIG_PATH, METRICS_PATH, VALIDATION_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load the exact category order and feature layout saved by notebook 15.
    print("\\nLoading the exact training preprocessing configuration...")
    with open(PREPROCESSOR_CONFIG_PATH, "r", encoding="utf-8") as handle:
        preprocessing_config = json.load(handle)

    categories = [preprocessing_config["categorical_categories"]]
    expected_feature_names = preprocessing_config["transformed_feature_names"]
    if preprocessing_config["numerical_features"] != NUMERICAL_FEATURES:
        raise ValueError("Numerical feature order differs from the training configuration.")

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
        verbose_feature_names_out=False,
    )

    # Fit only to establish the configured transformer structure; categories are fixed.
    category_frame = pd.DataFrame({
        "type": pd.Categorical(categories[0], categories=categories[0]),
        **{
            feature: np.zeros(len(categories[0]), dtype=np.float32)
            for feature in NUMERICAL_FEATURES
        },
    })
    encoder.fit(category_frame)
    feature_names = encoder.get_feature_names_out().tolist()
    if feature_names != expected_feature_names:
        raise ValueError("Reconstructed feature order does not match training metadata.")

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    expected_features = model.get_booster().num_features()
    if len(feature_names) != expected_features:
        raise ValueError(
            f"Feature mismatch: encoder produces {len(feature_names)} "
            f"features but model expects {expected_features}. "
            "Check the feature definitions against notebook 15."
        )

    print("Transformed features:", feature_names)

    print("\nLoading validation data...")
    validation = pd.read_csv(
        VALIDATION_PATH,
        dtype=DTYPES,
        usecols=FEATURES + [TARGET],
    )
    if validation.isnull().any().any():
        raise ValueError("Validation data contains missing values.")

    # Random sample for representative global feature attribution.
    sample_n = min(GLOBAL_SAMPLE_SIZE, len(validation))
    global_sample = validation.sample(
        n=sample_n, random_state=RANDOM_STATE
    ).copy()

    X_global = to_float32_csr(
        encoder.transform(global_sample[FEATURES])
    )
    y_global = global_sample[TARGET].to_numpy(dtype=np.int8)

    # Pick one correctly labelled fraud and one legitimate transaction
    # with the highest model scores for local case explanations.
    print("\nGenerating predictions for local examples...")
    # Preserve the CSR sparse representation used during benchmark training
    # and validation. Densifying changes implicit sparse zeros into explicit
    # numeric zeros, which can change XGBoost's missing-value routing.
    all_X = to_float32_csr(encoder.transform(validation[FEATURES]))
    print(
        "Validation matrix:",
        all_X.shape,
        "| format:",
        all_X.getformat(),
        "| dtype:",
        all_X.dtype,
    )
    all_probabilities = model.predict_proba(all_X)[:, 1]
    all_y = validation[TARGET].to_numpy(dtype=np.int8)

    # Verify that the loaded model and reconstructed inputs reproduce notebook 15.
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        benchmark = json.load(handle)

    reproduced_pr_auc = float(average_precision_score(all_y, all_probabilities))
    reproduced_roc_auc = float(roc_auc_score(all_y, all_probabilities))
    precision, recall, thresholds = precision_recall_curve(all_y, all_probabilities)
    f1_denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        f1_denominator,
        out=np.zeros_like(f1_denominator),
        where=f1_denominator > 0,
    )
    reproduced_threshold = float(thresholds[int(np.argmax(f1_values))])

    expected_validation = benchmark["validation"]
    checks = {
        "PR-AUC": (reproduced_pr_auc, float(expected_validation["pr_auc"])),
        "ROC-AUC": (reproduced_roc_auc, float(expected_validation["roc_auc"])),
        "F1 threshold": (reproduced_threshold, float(benchmark["selected_threshold"])),
    }
    print("\\n========== BENCHMARK CONSISTENCY CHECK ==========")
    for name, (actual, expected) in checks.items():
        print(f"{name}: reproduced={actual:.9f} | benchmark={expected:.9f}")
        if not np.isclose(actual, expected, rtol=1e-5, atol=1e-7):
            raise ValueError(
                f"{name} mismatch. The model, validation data, or preprocessing "
                "does not match the benchmark report. Do not interpret SHAP outputs."
            )
    print("PASS: SHAP predictions reproduce the benchmark validation results.")


    fraud_indices = np.flatnonzero(all_y == 1)
    legitimate_indices = np.flatnonzero(all_y == 0)
    if len(fraud_indices) == 0 or len(legitimate_indices) == 0:
        raise ValueError("Validation data must contain both classes.")

    fraud_index = fraud_indices[np.argmax(all_probabilities[fraud_indices])]
    legitimate_index = legitimate_indices[
        np.argmax(all_probabilities[legitimate_indices])
    ]

    X_local = all_X[[fraud_index, legitimate_index]].tocsr()
    local_labels = ["highest_scored_validation_fraud", "highest_scored_validation_legitimate"]
    local_probabilities = all_probabilities[[fraud_index, legitimate_index]]

    print("Validation sample rows:", len(global_sample))
    print("Validation fraud prevalence in global sample:", f"{y_global.mean():.6%}")
    print("Local case predicted probabilities:")
    for label, probability in zip(local_labels, local_probabilities):
        print(f"  {label}: {probability:.6f}")

    del all_X, all_probabilities, validation
    gc.collect()

    # TreeExplainer uses the fitted tree structure; no background training
    # dataset is required for this tree-based model.
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    global_explanation = explainer(X_global)
    local_explanation = explainer(X_local)

    # Global importance: mean absolute SHAP value.
    mean_abs_shap = np.abs(global_explanation.values).mean(axis=0)
    importance = sorted(
        [
            {"feature": name, "mean_abs_shap": float(value)}
            for name, value in zip(feature_names, mean_abs_shap)
        ],
        key=lambda item: item["mean_abs_shap"],
        reverse=True,
    )

    importance_path = REPORT_DIR / "xgboost_shap_feature_importance.json"
    with open(importance_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": "XGBoost no-ratio",
                "explanation_data": "random validation sample",
                "sample_size": int(sample_n),
                "random_state": RANDOM_STATE,
                "features": importance,
            },
            handle,
            indent=4,
        )

    # SHAP summary plots.
    plt.figure()
    shap.plots.bar(global_explanation, max_display=14, show=False)
    plt.tight_layout()
    bar_path = FIGURE_DIR / "xgboost_shap_global_bar.png"
    plt.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.beeswarm(global_explanation, max_display=14, show=False)
    plt.tight_layout()
    beeswarm_path = FIGURE_DIR / "xgboost_shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=180, bbox_inches="tight")
    plt.close()

    # Local waterfall plots: one fraud and one legitimate example.
    local_cases = []
    for index, label in enumerate(local_labels):
        plt.figure()
        shap.plots.waterfall(
            local_explanation[index], max_display=14, show=False
        )
        plt.tight_layout()
        plot_path = FIGURE_DIR / f"{label}_waterfall.png"
        plt.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close()

        local_cases.append(
            {
                "case": label,
                "actual_label": int(index == 0),
                "predicted_probability": float(local_probabilities[index]),
                "waterfall_plot": str(plot_path.relative_to(ROOT)),
            }
        )

    summary = {
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "explanation_data": "validation split only",
        "global_sample_size": int(sample_n),
        "random_state": RANDOM_STATE,
        "feature_names": feature_names,
        "global_mean_absolute_shap": importance,
        "local_examples": local_cases,
        "artifacts": {
            "feature_importance_json": str(importance_path.relative_to(ROOT)),
            "global_bar_plot": str(bar_path.relative_to(ROOT)),
            "beeswarm_plot": str(beeswarm_path.relative_to(ROOT)),
        },
        "note": (
            "SHAP explains model behaviour; it does not establish causality "
            "or prove that a feature is free from dataset-specific bias."
        ),
    }

    summary_path = REPORT_DIR / "xgboost_shap_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)

    print("\n========== SHAP ANALYSIS COMPLETE ==========")
    print("\nTop features by mean absolute SHAP:")
    for item in importance[:10]:
        print(f"  {item['feature']}: {item['mean_abs_shap']:.6f}")

    print("\nSaved artifacts:")
    print(importance_path)
    print(summary_path)
    print(bar_path)
    print(beeswarm_path)
    for case in local_cases:
        print(ROOT / case["waterfall_plot"])

    print(
        "\nNote: explanations use validation transactions only. "
        "No threshold tuning or test-set evaluation is performed."
    )


if __name__ == "__main__":
    main()
