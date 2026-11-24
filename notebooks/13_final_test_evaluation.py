from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Final evaluation ONLY. Do not tune thresholds or fit preprocessing here.
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

SPLIT_DIR = ROOT / "data" / "splits"
PREPROCESSOR_DIR = ROOT / "data" / "artifacts" / "preprocessing"
MODEL_DIR = ROOT / "models" / "baseline"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEST_PATH = SPLIT_DIR / "test.csv"
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
    "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio",
]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

DTYPES = {"type": "category", TARGET: "int8"}
DTYPES.update({name: "float32" for name in NUMERICAL_FEATURES})


def as_float32(matrix):
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def evaluate(name, model_path, preprocessor_path, validation_metrics_path):
    print(f"\n========== {name.upper()} FINAL TEST ==========")

    for path in (model_path, preprocessor_path, validation_metrics_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact missing: {path}\n"
                "Run the existing preprocessing and model-training scripts first."
            )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    with open(validation_metrics_path, "r", encoding="utf-8") as handle:
        validation_metrics = json.load(handle)

    # Threshold was selected on validation data; keep it fixed for test.
    threshold = float(validation_metrics["selected_threshold"])

    df = pd.read_csv(
        TEST_PATH,
        dtype=DTYPES,
        usecols=FEATURES + [TARGET, "step"],
    )

    if df.isnull().any().any():
        raise ValueError("Test split contains missing values.")

    y_test = df[TARGET].to_numpy(dtype=np.int8)
    if len(np.unique(y_test)) != 2:
        raise ValueError("Test split must contain both target classes.")

    print("Test rows:", f"{len(df):,}")
    print("Test fraud cases:", f"{int(y_test.sum()):,}")
    print("Test fraud prevalence:", f"{y_test.mean():.6%}")
    print("Test step range:", int(df["step"].min()), "-", int(df["step"].max()))
    print("Fixed validation threshold:", threshold)

    X_test_raw = df[FEATURES]
    del df
    gc.collect()

    X_test = as_float32(preprocessor.transform(X_test_raw))
    del X_test_raw
    gc.collect()

    values = X_test.data if sparse.issparse(X_test) else X_test
    if not np.isfinite(values).all():
        raise ValueError("Transformed test matrix contains non-finite values.")

    y_prob = model.predict_proba(X_test)[:, 1]
    del X_test
    gc.collect()

    y_pred = (y_prob >= threshold).astype(np.int8)

    results = {
        "model": name,
        "evaluation_split": "chronological_test",
        "threshold": threshold,
        "threshold_selected_on": "validation",
        "test_rows": int(len(y_test)),
        "test_fraud_cases": int(y_test.sum()),
        "test_fraud_prevalence": float(y_test.mean()),
        "pr_auc_average_precision": float(average_precision_score(y_test, y_prob)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix_labels": ["legitimate", "fraud"],
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "versions": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    print(f"PR-AUC:   {results['pr_auc_average_precision']:.6f}")
    print(f"ROC-AUC:  {results['roc_auc']:.6f}")
    print(f"Precision:{results['precision']:.6f}")
    print(f"Recall:   {results['recall']:.6f}")
    print(f"F1:       {results['f1_score']:.6f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(np.asarray(results["confusion_matrix"]))
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    ))

    output = REPORT_DIR / f"{name.lower().replace(' ', '_')}_test_metrics.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)

    print("Saved:", output)
    return results


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"{TEST_PATH} not found. Run notebooks/07_chronological_split.py first."
        )

    all_results = []

    all_results.append(evaluate(
        "Logistic Regression",
        MODEL_DIR / "logistic_regression.joblib",
        PREPROCESSOR_DIR / "logistic_preprocessor.joblib",
        REPORT_DIR / "logistic_regression_metrics.json",
    ))

    all_results.append(evaluate(
        "Random Forest",
        MODEL_DIR / "random_forest.joblib",
        PREPROCESSOR_DIR / "tree_preprocessor.joblib",
        REPORT_DIR / "random_forest_metrics.json",
    ))

    summary = {
        "evaluation_split": "chronological_test",
        "warning": "Thresholds were selected on validation data and were not retuned on test data.",
        "models": all_results,
    }

    output = REPORT_DIR / "final_test_comparison.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)

    print("\n========== FINAL TEST COMPARISON ==========")
    for result in all_results:
        print(
            f"{result['model']}: "
            f"PR-AUC={result['pr_auc_average_precision']:.6f}, "
            f"ROC-AUC={result['roc_auc']:.6f}, "
            f"Precision={result['precision']:.6f}, "
            f"Recall={result['recall']:.6f}, "
            f"F1={result['f1_score']:.6f}"
        )
    print("Comparison saved:", output)


if __name__ == "__main__":
    main()
