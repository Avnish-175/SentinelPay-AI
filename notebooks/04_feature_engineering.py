
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# SENTINELPAY AI - FEATURE ENGINEERING
# ==========================================

DATA_PATH = Path(
    r"C:\Users\ak789\Desktop\proj\SentinelPay-AI\data\raw\PS_20174392719_1491204439457_log.csv"
)

OUTPUT_PATH = Path(
    r"C:\Users\ak789\Desktop\proj\SentinelPay-AI\data\processed\sentinelpay_features.csv"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500_000

if not DATA_PATH.exists():
    raise FileNotFoundError(DATA_PATH)

if OUTPUT_PATH.exists():
    OUTPUT_PATH.unlink()

total_rows = 0
fraud_count = 0
legitimate_count = 0
first_chunk = True

print("Starting SentinelPay AI Feature Engineering...")

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1
):

    # 1. Transaction timing features
    chunk["transaction_day"] = (chunk["step"] - 1) // 24
    chunk["transaction_hour"] = (chunk["step"] - 1) % 24

    chunk["hour_sin"] = np.sin(
        2 * np.pi * chunk["transaction_hour"] / 24
    )

    chunk["hour_cos"] = np.cos(
        2 * np.pi * chunk["transaction_hour"] / 24
    )

    # 2. Balance change features
    chunk["origin_balance_change"] = (
        chunk["oldbalanceOrg"] - chunk["newbalanceOrig"]
    )

    chunk["destination_balance_change"] = (
        chunk["newbalanceDest"] - chunk["oldbalanceDest"]
    )

    # 3. Signed balance discrepancies
    chunk["origin_balance_error_signed"] = (
        chunk["oldbalanceOrg"]
        - chunk["amount"]
        - chunk["newbalanceOrig"]
    )

    chunk["destination_balance_error_signed"] = (
        chunk["oldbalanceDest"]
        + chunk["amount"]
        - chunk["newbalanceDest"]
    )

    # 4. Absolute balance discrepancies
    chunk["origin_balance_error"] = (
        chunk["origin_balance_error_signed"].abs()
    )

    chunk["destination_balance_error"] = (
        chunk["destination_balance_error_signed"].abs()
    )

    # 5. Zero balance indicators
    chunk["origin_is_zero"] = (
        chunk["oldbalanceOrg"] == 0
    ).astype("int8")

    chunk["destination_is_zero"] = (
        chunk["oldbalanceDest"] == 0
    ).astype("int8")

    # 6. Transaction amount ratios
    chunk["amount_to_origin_balance_ratio"] = np.where(
        chunk["oldbalanceOrg"] > 0,
        chunk["amount"] / chunk["oldbalanceOrg"],
        0
    )

    chunk["amount_to_destination_balance_ratio"] = np.where(
        chunk["oldbalanceDest"] > 0,
        chunk["amount"] / chunk["oldbalanceDest"],
        0
    )

    # 7. Remove identifiers and existing fraud flag
    chunk.drop(
        columns=[
            "nameOrig",
            "nameDest",
            "isFlaggedFraud"
        ],
        inplace=True
    )

    # 8. Validate target and features
    if chunk.isnull().any().any():
        raise ValueError(
            f"Missing values detected in chunk {chunk_number}"
        )

    total_rows += len(chunk)
    fraud_count += int(chunk["isFraud"].sum())
    legitimate_count += int(
        (chunk["isFraud"] == 0).sum()
    )

    # 9. Write processed chunk
    chunk.to_csv(
        OUTPUT_PATH,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False
    )

    first_chunk = False

    print(
        f"Processed chunk {chunk_number} | "
        f"Rows processed: {total_rows:,}"
    )

# ==========================================
# FINAL VALIDATION
# ==========================================

print("\n========== FEATURE ENGINEERING REPORT ==========")

print("Total processed rows:", f"{total_rows:,}")
print("Fraudulent transactions:", f"{fraud_count:,}")
print("Legitimate transactions:", f"{legitimate_count:,}")

print(
    "Fraud percentage:",
    round(fraud_count / total_rows * 100, 4),
    "%"
)

print("Output file:", OUTPUT_PATH)
print(
    "Output size:",
    round(OUTPUT_PATH.stat().st_size / (1024 ** 2), 2),
    "MB"
)

print("\nFeature columns:")
print(list(chunk.columns))

print("\nFeature engineering completed successfully!")
