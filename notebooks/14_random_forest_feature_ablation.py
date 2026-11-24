from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

TARGET = "isFraud"
CATEGORICAL = ["type"]
NUMERICAL = [
    "amount", "oldbalanceOrg", "oldbalanceDest",
    "transaction_day", "transaction_hour", "hour_sin", "hour_cos",
    "origin_is_zero", "destination_is_zero",
    "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio",
]
ALL_FEATURES = CATEGORICAL + NUMERICAL

DTYPES = {"type": "category", TARGET: "int8"}
DTYPES.update({name: "float32" for name in NUMERICAL})

VARIANTS = {
    "without_origin_ratio": [
        name for name in ALL_FEATURES
        if name != "amount_to_origin_balance_ratio"
    ],
    "without_any_ratio": [
        name for name in ALL_FEATURES
        if name not in {
            "amount_to_origin_balance_ratio",
            "amount_to_destination_balance_ratio",
        }
    ],
}


def load_xy(split, features):
    df = pd.read_csv(
        SPLITS / f"{split}.csv",
        dtype=DTYPES,
        usecols=features + [TARGET],
    )
    if df.isnull().any().any():
        raise ValueError(f"{split}: missing values detected")
    y = df[TARGET].to_numpy(dtype=np.int8)
    return df[features], y


def as_float32(matrix):
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def select_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def evaluate_variant(name, features):
    print(f"\n========== {name.upper()} ==========")
    categorical = [f for f in CATEGORICAL if f in features]
    numerical = [f for f in NUMERICAL if f in features]

    transformers = []
    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
            categorical,
        ))
    if numerical:
        transformers.append(("numerical", "passthrough", numerical))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )

    X_train_raw, y_train = load_xy("train", features)
    X_val_raw, y_val = load_xy("validation", features)

    preprocessor.fit(X_train_raw)
    X_train = as_float32(preprocessor.transform(X_train_raw))
    X_val = as_float32(preprocessor.transform(X_val_raw))
    del X_train_raw, X_val_raw
    gc.collect()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=10,
        max_features="sqrt",
        max_samples=0.25,
        bootstrap=True,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    print(f"Training seconds: {time.time() - start:.1f}")
    del X_train, y_train
    gc.collect()

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_prob)
    val_pr_auc = average_precision_score(y_val, val_prob)
    del X_val, val_prob, y_val
    gc.collect()

    X_test_raw, y_test = load_xy("test", features)
    X_test = as_float32(preprocessor.transform(X_test_raw))
    del X_test_raw
    gc.collect()

    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(np.int8)

    result = {
        "variant": name,
        "features": features,
        "threshold_selected_on": "validation",
        "selected_threshold": threshold,
        "validation_pr_auc": float(val_pr_auc),
        "test_rows": int(len(y_test)),
        "test_fraud_cases": int(y_test.sum()),
        "test_pr_auc": float(average_precision_score(y_test, test_prob)),
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist(),
    }

    print(f"Validation PR-AUC: {result['validation_pr_auc']:.6f}")
    print(f"Fixed threshold:   {threshold:.6f}")
    print(f"Test PR-AUC:       {result['test_pr_auc']:.6f}")
    print(f"Test ROC-AUC:      {result['test_roc_auc']:.6f}")
    print(f"Test precision:    {result['test_precision']:.6f}")
    print(f"Test recall:       {result['test_recall']:.6f}")
    print(f"Test F1:           {result['test_f1']:.6f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(np.asarray(result["test_confusion_matrix"]))

    del X_test, test_prob, test_pred, model, preprocessor
    gc.collect()
    return result


def main():
    results = []
    for name, features in VARIANTS.items():
        results.append(evaluate_variant(name, features))

    output = REPORTS / "random_forest_feature_ablation.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)

    print("\n========== ABLATION SUMMARY ==========")
    for result in results:
        print(
            f"{result['variant']}: "
            f"PR-AUC={result['test_pr_auc']:.6f}, "
            f"ROC-AUC={result['test_roc_auc']:.6f}, "
            f"Precision={result['test_precision']:.6f}, "
            f"Recall={result['test_recall']:.6f}, "
            f"F1={result['test_f1']:.6f}"
        )
    print("Saved:", output)


if __name__ == "__main__":
    main()
