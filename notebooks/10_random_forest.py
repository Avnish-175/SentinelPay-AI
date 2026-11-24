
from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn

from scipy import sparse

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "preprocessing"
    / "tree_preprocessor.joblib"
)

MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"

MODEL_PATH = MODEL_DIR / "random_forest.joblib"
METRICS_PATH = REPORT_DIR / "random_forest_metrics.json"


# ==========================================
# 2. FEATURE CONFIGURATION
# ==========================================

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

TARGET = "isFraud"

DTYPES = {
    "type": "category",
    TARGET: "int8",
}

DTYPES.update({
    feature: "float32"
    for feature in NUMERICAL_FEATURES
})


# ==========================================
# 3. MODEL CONFIGURATION
# ==========================================

N_ESTIMATORS = 100
MAX_DEPTH = 16
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 10
MAX_FEATURES = "sqrt"
MAX_SAMPLES = 0.25

CLASS_WEIGHT = "balanced_subsample"
RANDOM_STATE = 42
N_JOBS = -1


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def to_float32_matrix(matrix):

    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)

    return np.asarray(matrix, dtype=np.float32)


def is_finite(matrix):

    data = (
        matrix.data
        if sparse.issparse(matrix)
        else matrix
    )

    return bool(np.isfinite(data).all())


def load_split(path, name):

    df = pd.read_csv(
        path,
        dtype=DTYPES,
        usecols=FEATURES + [TARGET]
    )

    if df.isnull().any().any():
        raise ValueError(
            f"{name}: Missing values detected."
        )

    y = df[TARGET].to_numpy()

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"{name}: Invalid target values."
        )

    X = df[FEATURES]

    print(f"{name} records: {len(df):,}")
    print(f"{name} fraud cases: {int(y.sum()):,}")

    del df

    return X, y


def compute_metrics(y_true, y_prob, threshold):

    y_pred = (
        y_prob >= threshold
    ).astype(np.int8)

    metrics = {
        "threshold": float(threshold),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).tolist()
    }

    return metrics, y_pred


# ==========================================
# 5. LOAD PREPROCESSOR
# ==========================================

print("\n========== LOADING PREPROCESSOR ==========")

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(PREPROCESSOR_PATH)

preprocessor = joblib.load(PREPROCESSOR_PATH)

EXPECTED_N_FEATURES = len(
    preprocessor.get_feature_names_out()
)

FEATURE_NAMES = (
    preprocessor.get_feature_names_out().tolist()
)

print("Tree preprocessor loaded.")
print("Output features:", EXPECTED_N_FEATURES)


# ==========================================
# 6. LOAD TRAINING DATA
# ==========================================

print("\n========== LOADING TRAINING DATA ==========")

X_train_raw, y_train = load_split(
    TRAIN_PATH,
    "Training"
)

print("\n========== TRANSFORMING TRAINING DATA ==========")

X_train = to_float32_matrix(
    preprocessor.transform(X_train_raw)
)

print("Training matrix shape:", X_train.shape)
print("Training matrix type:", type(X_train).__name__)

assert X_train.shape[1] == EXPECTED_N_FEATURES

assert is_finite(X_train), (
    "Non-finite values in training matrix."
)

assert len(np.unique(y_train)) == 2, (
    "Training data must contain both classes."
)

del X_train_raw
gc.collect()


# ==========================================
# 7. INITIALIZE RANDOM FOREST
# ==========================================

print("\n========== INITIALIZING RANDOM FOREST ==========")

model = RandomForestClassifier(

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_split=MIN_SAMPLES_SPLIT,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    max_features=MAX_FEATURES,

    max_samples=MAX_SAMPLES,

    bootstrap=True,

    class_weight=CLASS_WEIGHT,

    random_state=RANDOM_STATE,

    n_jobs=N_JOBS,

    verbose=1
)

print("Algorithm: Random Forest")

print("Trees:", N_ESTIMATORS)
print("Maximum depth:", MAX_DEPTH)
print("Minimum samples leaf:", MIN_SAMPLES_LEAF)
print("Maximum sample fraction:", MAX_SAMPLES)
print("Class weight:", CLASS_WEIGHT)
print("Parallel jobs:", N_JOBS)


# ==========================================
# 8. TRAIN RANDOM FOREST
# ==========================================

print("\n========== MODEL TRAINING STARTED ==========")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("\nTraining completed.")

print(
    "Training time:",
    round(training_time, 2),
    "seconds"
)

print("Trees trained:", len(model.estimators_))

del X_train
del y_train
gc.collect()


# ==========================================
# 9. LOAD VALIDATION DATA
# ==========================================

print("\n========== LOADING VALIDATION DATA ==========")

X_val_raw, y_validation = load_split(
    VALIDATION_PATH,
    "Validation"
)

assert len(np.unique(y_validation)) == 2, (
    "Validation data must contain both classes."
)


# ==========================================
# 10. TRANSFORM VALIDATION DATA
# ==========================================

print("\n========== TRANSFORMING VALIDATION DATA ==========")

X_validation = to_float32_matrix(
    preprocessor.transform(X_val_raw)
)

print("Validation matrix shape:", X_validation.shape)

assert X_validation.shape[1] == EXPECTED_N_FEATURES

assert is_finite(X_validation), (
    "Non-finite values in validation matrix."
)

del X_val_raw
gc.collect()


# ==========================================
# 11. GENERATE PREDICTIONS
# ==========================================

print("\n========== GENERATING PREDICTIONS ==========")

start_time = time.time()

y_prob = model.predict_proba(
    X_validation
)[:, 1]

prediction_time = time.time() - start_time

print(
    "Prediction time:",
    round(prediction_time, 2),
    "seconds"
)

del X_validation
gc.collect()


# ==========================================
# 12. MODEL EVALUATION
# ==========================================

print("\n========== MODEL EVALUATION ==========")

pr_auc = average_precision_score(
    y_validation,
    y_prob
)

roc_auc = roc_auc_score(
    y_validation,
    y_prob
)

print(f"\nPR-AUC (Average Precision): {pr_auc:.6f}")
print(f"ROC-AUC: {roc_auc:.6f}")


# ------------------------------------------
# A. DEFAULT THRESHOLD
# ------------------------------------------

DEFAULT_THRESHOLD = 0.50

default_metrics, y_pred_default = compute_metrics(
    y_validation,
    y_prob,
    DEFAULT_THRESHOLD
)

print(f"\n--- Threshold = {DEFAULT_THRESHOLD} ---")

print(f"Precision: {default_metrics['precision']:.6f}")
print(f"Recall:    {default_metrics['recall']:.6f}")
print(f"F1 Score:  {default_metrics['f1_score']:.6f}")

print("\nConfusion Matrix:")

print(
    np.array(
        default_metrics["confusion_matrix"]
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        y_pred_default,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)


# ------------------------------------------
# B. BEST F1 THRESHOLD
# ------------------------------------------

precisions, recalls, thresholds = (
    precision_recall_curve(
        y_validation,
        y_prob
    )
)

if len(thresholds) == 0:
    raise ValueError(
        "No classification thresholds available."
    )

denominator = precisions[:-1] + recalls[:-1]

f1_scores = np.divide(
    2 * precisions[:-1] * recalls[:-1],
    denominator,
    out=np.zeros_like(denominator),
    where=denominator > 0
)

best_idx = int(np.argmax(f1_scores))

selected_threshold = float(
    thresholds[best_idx]
)

tuned_metrics, y_pred_tuned = compute_metrics(
    y_validation,
    y_prob,
    selected_threshold
)

print(
    f"\n--- Best-F1 threshold: "
    f"{selected_threshold:.6f} ---"
)

print(f"Precision: {tuned_metrics['precision']:.6f}")
print(f"Recall:    {tuned_metrics['recall']:.6f}")
print(f"F1 Score:  {tuned_metrics['f1_score']:.6f}")

print("\nConfusion Matrix:")

print(
    np.array(
        tuned_metrics["confusion_matrix"]
    )
)

print(
    "\nNOTE: Threshold selected on validation data. "
    "Its performance may be optimistic. "
    "Keep the selected threshold unchanged for final testing."
)


# ==========================================
# 13. FEATURE IMPORTANCE
# ==========================================

print("\n========== FEATURE IMPORTANCE ==========")

importances = model.feature_importances_

importance_df = pd.DataFrame({

    "feature": FEATURE_NAMES,

    "importance": importances

}).sort_values(
    by="importance",
    ascending=False
)

print("\nTop 10 important features:")

print(
    importance_df.head(10).to_string(index=False)
)

feature_importance_path = (
    REPORT_DIR / "random_forest_feature_importance.csv"
)

importance_df.to_csv(
    feature_importance_path,
    index=False
)


# ==========================================
# 14. SAVE MODEL AND METRICS
# ==========================================

print("\n========== SAVING ARTIFACTS ==========")

metrics = {

    "model": "Random Forest",

    "hyperparameters": {

        "n_estimators": N_ESTIMATORS,

        "max_depth": MAX_DEPTH,

        "min_samples_split": MIN_SAMPLES_SPLIT,

        "min_samples_leaf": MIN_SAMPLES_LEAF,

        "max_features": MAX_FEATURES,

        "max_samples": MAX_SAMPLES,

        "class_weight": CLASS_WEIGHT,

        "bootstrap": True,

        "random_state": RANDOM_STATE,

        "n_jobs": N_JOBS
    },

    "training_time_seconds": round(
        training_time, 2
    ),

    "prediction_time_seconds": round(
        prediction_time, 2
    ),

    "trees_trained": len(model.estimators_),

    "pr_auc_average_precision": float(pr_auc),

    "roc_auc": float(roc_auc),

    "default_threshold_metrics": default_metrics,

    "best_f1_threshold_metrics": tuned_metrics,

    "selected_threshold": selected_threshold,

    "threshold_selected_on": "validation",

    "features": FEATURES,

    "transformed_feature_count": EXPECTED_N_FEATURES,

    "transformed_feature_names": FEATURE_NAMES,

    "top_10_features": (
        importance_df.head(10).to_dict(orient="records")
    ),

    "versions": {

        "sklearn": sklearn.__version__,

        "numpy": np.__version__,

        "pandas": pd.__version__
    }
}


joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w") as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ==========================================
# 15. FINAL SUMMARY
# ==========================================

print("\n========== RANDOM FOREST COMPLETED ==========")

print("Model:", MODEL_PATH)

print("Metrics:", METRICS_PATH)

print("Feature importance:", feature_importance_path)

print("PR-AUC:", round(pr_auc, 6))

print("ROC-AUC:", round(roc_auc, 6))

print("Selected threshold:", selected_threshold)

print("\nRandom Forest baseline saved successfully!")
