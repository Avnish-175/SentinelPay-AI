
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "splits" / "validation.csv"

df = pd.read_csv(PATH)

# Direct comparison
df["direct_balance_equal"] = np.isclose(
    df["amount"],
    df["oldbalanceOrg"],
    rtol=1e-5,
    atol=0.01
)

# Existing engineered ratio comparison
df["ratio_equal"] = np.isclose(
    df["amount_to_origin_balance_ratio"],
    1.0
)

print("\n========== FEATURE COMPARISON ==========")

print(pd.crosstab(
    df["ratio_equal"],
    df["direct_balance_equal"],
    margins=True
))

print("\n========== MISMATCHED TRANSACTIONS ==========")

mismatch = df[
    df["ratio_equal"] != df["direct_balance_equal"]
]

print("Mismatch count:", len(mismatch))

print(
    mismatch[
        [
            "type",
            "amount",
            "oldbalanceOrg",
            "amount_to_origin_balance_ratio",
            "isFraud",
            "ratio_equal",
            "direct_balance_equal"
        ]
    ].head(20).to_string(index=False)
)

print("\n========== FRAUD IN MISMATCHED ROWS ==========")

print(
    mismatch["isFraud"].value_counts()
)

print("\n========== RATIO FEATURE STATISTICS ==========")

print(
    df["amount_to_origin_balance_ratio"].describe()
)

print("\n========== ANALYSIS COMPLETE ==========")
