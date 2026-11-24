
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sentinelpay_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. APPROVED FEATURES
# ==========================================

FEATURES = [
    "type",
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

METADATA = ["step"]
TARGET = "isFraud"

REQUIRED_COLUMNS = FEATURES + METADATA + [TARGET]


# ==========================================
# 3. LOAD DATASET
# ==========================================

print("\nLoading SentinelPay dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Original dataset shape: {df.shape}")

missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Retain only approved features, metadata and target.
# Post-transaction leakage features are discarded.

df = df[REQUIRED_COLUMNS].copy()

print(f"Selected dataset shape: {df.shape}")


# ==========================================
# 4. CHRONOLOGICAL SPLIT
# ==========================================

print("\nPerforming chronological split...")

unique_steps = np.sort(df["step"].unique())

total_steps = len(unique_steps)

train_index = int(total_steps * 0.70) - 1
validation_index = int(total_steps * 0.85) - 1

train_end_step = unique_steps[train_index]
validation_end_step = unique_steps[validation_index]

# Training: earliest 70% of time steps
train_df = df[
    df["step"] <= train_end_step
].copy()

# Validation: next 15% of time steps
validation_df = df[
    (df["step"] > train_end_step)
    & (df["step"] <= validation_end_step)
].copy()

# Testing: latest 15% of time steps
test_df = df[
    df["step"] > validation_end_step
].copy()


# ==========================================
# 5. VALIDATE SPLIT
# ==========================================

print("\n========== SPLIT VALIDATION ==========")

assert len(train_df) > 0
assert len(validation_df) > 0
assert len(test_df) > 0

assert train_df["step"].max() < validation_df["step"].min()

assert validation_df["step"].max() < test_df["step"].min()

assert (
    len(train_df)
    + len(validation_df)
    + len(test_df)
    == len(df)
)

print("Chronological ordering: PASSED")
print("No overlapping time periods: PASSED")
print("No transactions lost: PASSED")


# ==========================================
# 6. DATASET STATISTICS
# ==========================================

def display_statistics(name, dataset):

    total = len(dataset)
    fraud = int(dataset[TARGET].sum())
    legitimate = total - fraud

    fraud_percentage = (
        fraud / total * 100 if total else 0
    )

    print(f"\n{name}")
    print("-" * 45)

    print(f"Total transactions : {total:,}")
    print(f"Fraud transactions : {fraud:,}")
    print(f"Legitimate         : {legitimate:,}")

    print(f"Fraud percentage   : {fraud_percentage:.5f}%")

    print(
        f"Step range         : "
        f"{dataset['step'].min()} - "
        f"{dataset['step'].max()}"
    )


display_statistics("TRAINING DATA", train_df)

display_statistics("VALIDATION DATA", validation_df)

display_statistics("TESTING DATA", test_df)


# ==========================================
# 7. SAVE SPLITS
# ==========================================

print("\nSaving datasets...")

train_df.to_csv(
    OUTPUT_DIR / "train.csv",
    index=False
)

validation_df.to_csv(
    OUTPUT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    OUTPUT_DIR / "test.csv",
    index=False
)


# ==========================================
# 8. FINAL SUMMARY
# ==========================================

print("\n========== SPLIT COMPLETED ==========")

print(f"Total records: {len(df):,}")

print(f"Training records: {len(train_df):,}")

print(f"Validation records: {len(validation_df):,}")

print(f"Testing records: {len(test_df):,}")

print(f"\nOutput directory: {OUTPUT_DIR}")

print("\nAll chronological split operations completed!")
