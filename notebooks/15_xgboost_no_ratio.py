from pathlib import Path
import gc
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models" / "baseline"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

MODEL_PATH = MODEL_DIR / "xgboost_no_ratio.json"
METRICS_PATH = REPORT_DIR / "xgboost_no_ratio_metrics.json"
PREPROCESSOR_CONFIG_PATH = MODEL_DIR / "xgboost_no_ratio_preprocessing.json"

TARGET = "isFraud"
CATEGORICAL_FEATURES = ["type"]

# Deliberately exclude both engineered ratio features.
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

DTYPES = {
    "type": "category",
    TARGET: "int8",
}
DTYPES.update({feature: "float32" for feature in NUMERICAL_FEATURES})

# Fixed, documented baseline configuration. Do not tune on test.
PARAMETERS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# 2. HELPERS
# ============================================================

def load_split(path, split_name):
    df = pd.read_csv(
        path,
        dtype=DTYPES,
        usecols=FEATURES + [TARGET],
    )

    if df.isnull().any().any():
        raise ValueError(f"{split_name}: missing values detected")

    y = df[TARGET].to_numpy(dtype=np.int8)
    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{split_name}: target must contain only 0/1")

    X = df[FEATURES]
    print(
        f"{split_name}: {len(df):,} rows | "
        f"{int(y.sum()):,} fraud | "
        f"prevalence={y.mean():.6%}"
    )
    del df
    return X, y


def to_float32_csr(matrix):
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))


def select_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )
    if len(thresholds) == 0:
        raise ValueError("No thresholds returned from validation predictions")

    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f1_values))])


def classification_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int8)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def check_matrix(matrix, name):
    if matrix.shape[1] == 0:
        raise ValueError(f"{name}: transformed matrix has no features")
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"{name}: non-finite values found")


# ============================================================
# 3. LOAD AND FIT PREPROCESSOR ON TRAIN ONLY
# ============================================================

print("\n========== XGBOOST: NO-RATIO BENCHMARK ==========")
print("XGBoost version:", xgb.__version__)
print("Features:", FEATURES)

encoder = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
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

print("\n========== LOADING TRAINING DATA ==========")
X_train_raw, y_train = load_split(TRAIN_PATH, "Training")

if len(np.unique(y_train)) != 2:
    raise ValueError("Training split must contain both classes")

print("\nFitting categorical encoder on training data only...")
encoder.fit(X_train_raw)

feature_names = encoder.get_feature_names_out().tolist()
categorical_categories = (
    encoder.named_transformers_["categorical"].categories_[0].tolist()
)
with open(PREPROCESSOR_CONFIG_PATH, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "categorical_categories": categorical_categories,
            "numerical_features": NUMERICAL_FEATURES,
            "transformed_feature_names": feature_names,
        },
        handle,
        indent=4,
    )

X_train = to_float32_csr(encoder.transform(X_train_raw))
del X_train_raw
gc.collect()
check_matrix(X_train, "Training")

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
scale_pos_weight = negative_count / positive_count

print("Training matrix:", X_train.shape)
print("Negative training cases:", negative_count)
print("Positive training cases:", positive_count)
print("scale_pos_weight:", round(scale_pos_weight, 4))

# ============================================================
# 4. TRAIN XGBOOST
# ============================================================

model_parameters = dict(PARAMETERS)
model_parameters["scale_pos_weight"] = scale_pos_weight

model = xgb.XGBClassifier(**model_parameters)

print("\n========== TRAINING XGBOOST ==========")
print("Parameters:", json.dumps(model_parameters, indent=2))

start = time.time()
model.fit(X_train, y_train, verbose=False)
training_seconds = time.time() - start

print(f"Training completed in {training_seconds:.1f} seconds")
del X_train, y_train
gc.collect()

# ============================================================
# 5. SELECT THRESHOLD ON VALIDATION ONLY
# ============================================================

print("\n========== VALIDATION ==========")
X_val_raw, y_val = load_split(VALIDATION_PATH, "Validation")
X_val = to_float32_csr(encoder.transform(X_val_raw))
del X_val_raw
gc.collect()
check_matrix(X_val, "Validation")

val_probabilities = model.predict_proba(X_val)[:, 1]
validation_pr_auc = average_precision_score(y_val, val_probabilities)
validation_roc_auc = roc_auc_score(y_val, val_probabilities)
selected_threshold = select_f1_threshold(y_val, val_probabilities)
validation_metrics = classification_metrics(
    y_val, val_probabilities, selected_threshold
)

print(f"Validation PR-AUC: {validation_pr_auc:.6f}")
print(f"Validation ROC-AUC: {validation_roc_auc:.6f}")
print(f"Selected F1 threshold: {selected_threshold:.6f}")
print("Validation confusion matrix:")
print(np.asarray(validation_metrics["confusion_matrix_tn_fp_fn_tp"]))

del X_val, val_probabilities, y_val
gc.collect()

# ============================================================
# 6. FINAL EVALUATION ON CHRONOLOGICAL TEST SPLIT
# ============================================================

print("\n========== CHRONOLOGICAL TEST EVALUATION ==========")
X_test_raw, y_test = load_split(TEST_PATH, "Test")
X_test = to_float32_csr(encoder.transform(X_test_raw))
del X_test_raw
gc.collect()
check_matrix(X_test, "Test")

test_probabilities = model.predict_proba(X_test)[:, 1]
test_pr_auc = average_precision_score(y_test, test_probabilities)
test_roc_auc = roc_auc_score(y_test, test_probabilities)
test_metrics = classification_metrics(
    y_test, test_probabilities, selected_threshold
)

print(f"Test PR-AUC: {test_pr_auc:.6f}")
print(f"Test ROC-AUC: {test_roc_auc:.6f}")
print(f"Fixed validation threshold: {selected_threshold:.6f}")
print(f"Test precision: {test_metrics['precision']:.6f}")
print(f"Test recall: {test_metrics['recall']:.6f}")
print(f"Test F1: {test_metrics['f1']:.6f}")
print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
print(np.asarray(test_metrics["confusion_matrix_tn_fp_fn_tp"]))

# ============================================================
# 7. SAVE MODEL AND REPORT
# ============================================================

model.save_model(MODEL_PATH)

report = {
    "model": "XGBoost",
    "feature_set": "no_ratio",
    "features": FEATURES,
    "preprocessing_config": str(PREPROCESSOR_CONFIG_PATH.relative_to(ROOT)),
    "transformed_feature_names": feature_names,
    "hyperparameters": model_parameters,
    "training_seconds": round(training_seconds, 2),
    "training_class_counts": {
        "legitimate": negative_count,
        "fraud": positive_count,
    },
    "scale_pos_weight": float(scale_pos_weight),
    "threshold_selected_on": "validation",
    "selected_threshold": selected_threshold,
    "validation": {
        "pr_auc": float(validation_pr_auc),
        "roc_auc": float(validation_roc_auc),
        **validation_metrics,
    },
    "test": {
        "rows": int(len(y_test)),
        "fraud_cases": int(y_test.sum()),
        "fraud_prevalence": float(y_test.mean()),
        "pr_auc": float(test_pr_auc),
        "roc_auc": float(test_roc_auc),
        **test_metrics,
    },
    "xgboost_version": xgb.__version__,
    "sklearn_version": __import__("sklearn").__version__,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
}

with open(METRICS_PATH, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=4)

print("\n========== XGBOOST BENCHMARK COMPLETE ==========")
print("Model saved:", MODEL_PATH)
print("Metrics saved:", METRICS_PATH)
print("Test PR-AUC:", round(test_pr_auc, 6))
print("Test ROC-AUC:", round(test_roc_auc, 6))
print("Test F1:", round(test_metrics["f1"], 6))
print(
    "\nNote: the chronological test split has already been used in "
    "earlier model comparisons. Treat this result as diagnostic; reserve "
    "a fresh future-period holdout for a final unbiased estimate."
)
