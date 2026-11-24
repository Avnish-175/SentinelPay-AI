
import pandas as pd
from pathlib import Path

DATA_PATH = Path("C:\\Users\\ak789\\Desktop\\proj\\SentinelPay-AI\\data\\raw\\PS_20174392719_1491204439457_log.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

print("Starting SentinelPay AI dataset analysis...")

total_rows = 0
fraud_count = 0
legitimate_count = 0
missing_values = None
transaction_types = {}

for chunk in pd.read_csv(DATA_PATH, chunksize=500_000):
    total_rows += len(chunk)

    fraud_count += (chunk["isFraud"] == 1).sum()
    legitimate_count += (chunk["isFraud"] == 0).sum()

    current_missing = chunk.isnull().sum()

    if missing_values is None:
        missing_values = current_missing
    else:
        missing_values += current_missing

    counts = chunk["type"].value_counts()
    for transaction_type, count in counts.items():
        transaction_types[transaction_type] = (
            transaction_types.get(transaction_type, 0) + count
        )

print("\n========== DATASET REPORT ==========")
print("Total transactions:", total_rows)
print("Fraudulent transactions:", fraud_count)
print("Legitimate transactions:", legitimate_count)

print("\nFraud percentage:",
      round((fraud_count / total_rows) * 100, 4), "%")

print("\nMissing values:")
print(missing_values)

print("\nTransaction types:")
print(transaction_types)

print("\nDataset analysis completed!")
