
import pandas as pd
from pathlib import Path

DATA_PATH = Path(
    r"C:\Users\ak789\Desktop\proj\SentinelPay-AI\data\processed\sentinelpay_features.csv"
)

# Features available at authorization time
SAFE_FEATURES = [
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
    "amount_to_destination_balance_ratio"
]

# Features that depend on post-transaction balances
LEAKAGE_FEATURES = [
    "newbalanceOrig",
    "newbalanceDest",
    "origin_balance_change",
    "destination_balance_change",
    "origin_balance_error_signed",
    "destination_balance_error_signed",
    "origin_balance_error",
    "destination_balance_error"
]

TARGET = "isFraud"
METADATA = ["step"]

print("Starting SentinelPay AI Feature Leakage Audit...")

# Read only the CSV header; no need to load 1.17 GB
available_columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()

print("\n========== FEATURE AUDIT ==========")

missing_safe = [
    col for col in SAFE_FEATURES
    if col not in available_columns
]

missing_leakage = [
    col for col in LEAKAGE_FEATURES
    if col not in available_columns
]

if TARGET not in available_columns:
    raise ValueError("Target column is missing!")

if missing_safe:
    raise ValueError(f"Missing candidate features: {missing_safe}")

if missing_leakage:
    raise ValueError(f"Missing expected leakage columns: {missing_leakage}")

print("\n1. SAFE CANDIDATE FEATURES")
for feature in SAFE_FEATURES:
    print("[KEEP]", feature)

print("\n2. POST-TRANSACTION FEATURES")
for feature in LEAKAGE_FEATURES:
    print("[EXCLUDE]", feature)

print("\n3. SPLIT METADATA")
for feature in METADATA:
    print("[METADATA]", feature)

print("\n4. TARGET")
print("[TARGET]", TARGET)

print("\nTotal candidate features:", len(SAFE_FEATURES))
print("Total excluded leakage features:", len(LEAKAGE_FEATURES))

print("\n========== AUDIT COMPLETED ==========")
print("No post-transaction features will enter the baseline model.")
