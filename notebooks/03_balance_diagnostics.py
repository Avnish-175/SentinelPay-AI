
import pandas as pd
from pathlib import Path

DATA_PATH = Path(
    r"C:\Users\ak789\Desktop\proj\SentinelPay-AI\data\raw\PS_20174392719_1491204439457_log.csv"
)

stats = []

for chunk in pd.read_csv(DATA_PATH, chunksize=500_000):

    chunk["origin_error"] = (
        chunk["oldbalanceOrg"]
        - chunk["amount"]
        - chunk["newbalanceOrig"]
    ).abs()

    chunk["dest_error"] = (
        chunk["oldbalanceDest"]
        + chunk["amount"]
        - chunk["newbalanceDest"]
    ).abs()

    chunk["origin_zero"] = (
        chunk["oldbalanceOrg"] == 0
    )

    chunk["dest_zero"] = (
        chunk["oldbalanceDest"] == 0
    )

    grouped = chunk.groupby(["type", "isFraud"]).agg(
        count=("amount", "size"),
        avg_origin_error=("origin_error", "mean"),
        avg_dest_error=("dest_error", "mean"),
        median_origin_error=("origin_error", "median"),
        median_dest_error=("dest_error", "median"),
        origin_zero_count=("origin_zero", "sum"),
        dest_zero_count=("dest_zero", "sum")
    ).reset_index()

    stats.append(grouped)

# Combine chunk-level statistics correctly
df = pd.concat(stats, ignore_index=True)

weighted_columns = [
    "avg_origin_error",
    "avg_dest_error"
]

result = df.groupby(["type", "isFraud"]).apply(
    lambda group: pd.Series({
        "transactions": group["count"].sum(),
        "avg_origin_error": (
            group["avg_origin_error"] * group["count"]
        ).sum() / group["count"].sum(),
        "avg_dest_error": (
            group["avg_dest_error"] * group["count"]
        ).sum() / group["count"].sum(),
        "origin_zero_percent": (
            group["origin_zero_count"].sum()
            / group["count"].sum() * 100
        ),
        "dest_zero_percent": (
            group["dest_zero_count"].sum()
            / group["count"].sum() * 100
        )
    })
).reset_index()

print("\n========== BALANCE DIAGNOSTICS ==========")
print(result.round(3).to_string(index=False))

print("\n========== COMPLETED ==========")
