
from pathlib import Path
import json
import gc

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts" / "preprocessing"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

CHUNK_SIZE = 200_000

# ==========================================
# 2. FEATURE DEFINITIONS
# ==========================================

CATEGORICAL_FEATURES = [
    "type"
]

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
METADATA = ["step"]

REQUIRED_COLUMNS = FEATURES + METADATA + [TARGET]

# Use memory-efficient data types.
DTYPES = {
    "type": "category",
    "step": "int16",
    "isFraud": "int8",
}

for feature in NUMERICAL_FEATURES:
    DTYPES[feature] = "float32"


# ==========================================
# 3. DATA VALIDATION
# ==========================================

def validate_dataframe(df, dataset_name):

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name}: Missing columns: {missing}"
        )

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(
            f"{dataset_name}: Missing values detected"
        )

    numeric_data = df[NUMERICAL_FEATURES].to_numpy()

    if not np.isfinite(numeric_data).all():
        raise ValueError(
            f"{dataset_name}: Infinite values detected"
        )

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(
            f"{dataset_name}: Invalid target values"
        )

    print(f"{dataset_name}: Validation PASSED")


# ==========================================
# 4. LOAD TRAINING DATA
# ==========================================

print("\n========== LOADING TRAINING DATA ==========")

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype=DTYPES
)

print("Training shape:", train_df.shape)

validate_dataframe(train_df, "Training")


# ==========================================
# 5. DEFINE PREPROCESSING PIPELINES
# ==========================================

print("\n========== DEFINING PREPROCESSORS ==========")

# Logistic Regression:
# One-Hot Encoding + Standard Scaling

logistic_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                dtype=np.float32
            ),
            CATEGORICAL_FEATURES
        ),
        (
            "numerical",
            StandardScaler(),
            NUMERICAL_FEATURES
        )
    ],
    remainder="drop",
    sparse_threshold=1.0,
    verbose_feature_names_out=False
)


# Random Forest and XGBoost:
# One-Hot Encoding + Original Numerical Scales

tree_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
                dtype=np.float32
            ),
            CATEGORICAL_FEATURES
        ),
        (
            "numerical",
            "passthrough",
            NUMERICAL_FEATURES
        )
    ],
    remainder="drop",
    sparse_threshold=1.0,
    verbose_feature_names_out=False
)


# ==========================================
# 6. FIT ON TRAINING DATA ONLY
# ==========================================

print("\n========== FITTING PREPROCESSORS ==========")

X_train = train_df[FEATURES]

print("Fitting Logistic Regression preprocessor...")

logistic_preprocessor.fit(X_train)

print("Logistic Regression preprocessing fitted.")

print("\nFitting Tree-based preprocessor...")

tree_preprocessor.fit(X_train)

print("Tree-based preprocessing fitted.")

del X_train
del train_df

gc.collect()


# ==========================================
# 7. SAVE PREPROCESSING ARTIFACTS
# ==========================================

print("\n========== SAVING ARTIFACTS ==========")

logistic_path = (
    ARTIFACT_DIR / "logistic_preprocessor.joblib"
)

tree_path = (
    ARTIFACT_DIR / "tree_preprocessor.joblib"
)

joblib.dump(logistic_preprocessor, logistic_path)

joblib.dump(tree_preprocessor, tree_path)

logistic_features = (
    logistic_preprocessor.get_feature_names_out().tolist()
)

tree_features = (
    tree_preprocessor.get_feature_names_out().tolist()
)

with open(
    ARTIFACT_DIR / "feature_names.json",
    "w"
) as file:

    json.dump(
        {
            "logistic_regression": logistic_features,
            "tree_models": tree_features
        },
        file,
        indent=4
    )

print("Logistic preprocessor saved.")
print("Tree preprocessor saved.")
print("Feature names saved.")


# ==========================================
# 8. VALIDATE TRANSFORMATIONS
# ==========================================

def validate_transformation(
    file_path,
    dataset_name,
    preprocessor
):

    print(f"\nChecking {dataset_name}...")

    total_rows = 0
    expected_width = len(
        preprocessor.get_feature_names_out()
    )

    for chunk in pd.read_csv(
        file_path,
        dtype=DTYPES,
        chunksize=CHUNK_SIZE
    ):

        validate_dataframe(chunk, dataset_name)

        X = chunk[FEATURES]

        transformed = preprocessor.transform(X)

        if transformed.shape[1] != expected_width:
            raise ValueError(
                f"{dataset_name}: Unexpected feature count"
            )

        if not np.isfinite(transformed.data).all():
            raise ValueError(
                f"{dataset_name}: Invalid transformed values"
            )

        total_rows += len(chunk)

        del X
        del transformed

    print(f"{dataset_name} rows checked: {total_rows:,}")
    print("Transformation validation PASSED")


# Validate both preprocessing pipelines.
for name, path in [
    ("TRAINING", TRAIN_PATH),
    ("VALIDATION", VALIDATION_PATH),
    ("TESTING", TEST_PATH)
]:

    validate_transformation(
        path,
        name,
        logistic_preprocessor
    )

    validate_transformation(
        path,
        name,
        tree_preprocessor
    )


# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\n========== PREPROCESSING COMPLETED ==========")

print(
    "Logistic Regression features:",
    len(logistic_features)
)

print(
    "Tree-based model features:",
    len(tree_features)
)

print("Training-only fitting: PASSED")
print("Validation transformation: PASSED")
print("Testing transformation: PASSED")

print("\nSaved artifacts:")
print(logistic_path)
print(tree_path)
print(ARTIFACT_DIR / "feature_names.json")

print("\nAll preprocessing operations completed!")
