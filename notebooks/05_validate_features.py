
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(
    r"C:\Users\ak789\Desktop\proj\SentinelPay-AI\data\processed\sentinelpay_features.csv"
)

CHUNK_SIZE = 500_000

expected_rows = 6_362_620
total_rows = 0
fraud_count = 0
missing_count = 0
infinite_count = 0

min_values = {}
max_values = {}

print("Starting SentinelPay AI dataset validation...")

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE),
    start=1
):
    total_rows += len(chunk)

    fraud_count += int(chunk["isFraud"].sum())
    missing_count += int(chunk.isnull().sum().sum())

    numeric = chunk.select_dtypes(include="number")

    finite_mask = np.isfinite(numeric)
    infinite_count += int((~finite_mask).sum().sum())

    for column in numeric.columns:
        values = numeric[column].replace(
            [np.inf, -np.inf], np.nan
        )

        if values.notna().any():
            current_min = values.min()
            current_max = values.max()

            min_values[column] = min(
                min_values.get(column, current_min),
                current_min
            )

            max_values[column] = max(
                max_values.get(column, current_max),
                current_max
            )

    print(f"Validated chunk {chunk_number}")

print("\n========== VALIDATION REPORT ==========")

print("Total rows:", total_rows)
print("Expected rows:", expected_rows)
print("Row count matches:", total_rows == expected_rows)

print("Fraud count:", fraud_count)
print("Legitimate count:", total_rows - fraud_count)

print("Missing values:", missing_count)
print("Infinite values:", infinite_count)

print("\n========== FEATURE RANGES ==========")

for column in min_values:
    print(
        f"{column}: "
        f"Min={min_values[column]:.4f}, "
        f"Max={max_values[column]:.4f}"
    )

print("\n========== VALIDATION STATUS ==========")

if (
    total_rows == expected_rows
    and fraud_count == 8213
    and missing_count == 0
    and infinite_count == 0
):
    print("SUCCESS: Dataset passed basic validation!")
else:
    print("WARNING: Investigate validation failures.")

