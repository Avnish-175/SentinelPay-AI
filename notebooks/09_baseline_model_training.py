
from pathlib import Path
import gc
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from scipy import sparse

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

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
    / "logistic_preprocessor.joblib"
)

MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
METRICS_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"

MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"

METRICS_PATH = (
    METRICS_DIR / "logistic_regression_metrics.json"
)


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

SOLVER = "saga"
CLASS_WEIGHT = "balanced"

MAX_ITER = 1000
TOL = 1e-3

RANDOM_STATE = 42

DEFAULT_THRESHOLD = 0.50


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

print("Preprocessor loaded.")
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
# 7. INITIALIZE MODEL
# ==========================================

print("\n========== INITIALIZING MODEL ==========")

model = LogisticRegression(
    solver=SOLVER,
    class_weight=CLASS_WEIGHT,
    max_iter=MAX_ITER,
    tol=TOL,
    random_state=RANDOM_STATE,
)

print("Algorithm: Logistic Regression")

print(
    f"Solver: {SOLVER} | "
    f"Class weight: {CLASS_WEIGHT} | "
    f"Max iterations: {MAX_ITER} | "
    f"Tol: {TOL}"
)


# ==========================================
# 8. TRAIN MODEL
# ==========================================

print("\n========== MODEL TRAINING STARTED ==========")

start_time = time.time()

with warnings.catch_warnings(record=True) as caught_warnings:

    warnings.simplefilter("always")

    model.fit(X_train, y_train)

training_time = time.time() - start_time

converged = not any(
    issubclass(w.category, ConvergenceWarning)
    for w in caught_warnings
)

print("\nTraining completed.")

for warning in caught_warnings:

    print(
        f"Training warning "
        f"[{warning.category.__name__}]: "
        f"{warning.message}"
    )

print("Training time:", round(training_time, 2), "seconds")

print("Iterations:", int(model.n_iter_[0]))

print("Converged:", converged)

if not converged:

    raise RuntimeError(
        "Logistic Regression did not converge. "
        "Do not save this model as a successful baseline. "
        "Review convergence warnings and adjust training."
    )

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
# 10. GENERATE PREDICTIONS
# ==========================================

print("\n========== GENERATING PREDICTIONS ==========")

start_time = time.time()

y_prob = model.predict_proba(X_validation)[:, 1]

prediction_time = time.time() - start_time

print(
    "Prediction time:",
    round(prediction_time, 2),
    "seconds"
)

del X_validation
gc.collect()


# ==========================================
# 11. MODEL EVALUATION
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
    "\nNOTE: Threshold selected using validation data. "
    "Its performance is subject to validation overfitting. "
    "Keep this threshold unchanged for final test evaluation."
)


# ==========================================
# 12. SAVE MODEL AND METRICS
# ==========================================

print("\n========== SAVING ARTIFACTS ==========")

metrics = {

    "model": "Logistic Regression",

    "hyperparameters": {
        "solver": SOLVER,
        "class_weight": CLASS_WEIGHT,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "random_state": RANDOM_STATE,
    },

    "training_time_seconds": round(
        training_time, 2
    ),

    "prediction_time_seconds": round(
        prediction_time, 2
    ),

    "iterations": int(model.n_iter_[0]),

    "converged": converged,

    "pr_auc_average_precision": float(pr_auc),

    "roc_auc": float(roc_auc),

    "default_threshold_metrics": default_metrics,

    "best_f1_threshold_metrics": tuned_metrics,

    "selected_threshold": selected_threshold,

    "threshold_selected_on": "validation",

    "features": FEATURES,

    "transformed_feature_count": EXPECTED_N_FEATURES,

    "transformed_feature_names": (
        preprocessor.get_feature_names_out().tolist()
    ),

    "versions": {
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
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
# 13. FINAL SUMMARY
# ==========================================

print("\n========== BASELINE TRAINING COMPLETED ==========")

print("Model:", MODEL_PATH)

print("Metrics:", METRICS_PATH)

print("Converged:", converged)

print("PR-AUC:", round(pr_auc, 6))

print("ROC-AUC:", round(roc_auc, 6))

print("Selected threshold:", selected_threshold)

print("\nLogistic Regression baseline saved successfully!")
