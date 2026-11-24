
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = ROOT / "data/splits/validation.csv"

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

df = pd.read_csv(VALIDATION_PATH)

y = df["isFraud"]

print("\n========== VALIDATION DATA AUDIT ==========")

print("Shape:", df.shape)
print("\nTarget distribution:")
print(y.value_counts())

print("\nFraud prevalence:", y.mean())

print("\n========== INDIVIDUAL FEATURE ANALYSIS ==========")

results = []

for feature in FEATURES:

    values = df[feature]

    if feature == "type":
        continue

    if values.nunique() <= 1:
        continue

    ap = average_precision_score(y, values)
    roc = roc_auc_score(y, values)

    roc_reverse = roc_auc_score(y, -values)

    results.append({
        "feature": feature,
        "unique_values": values.nunique(),
        "average_precision": ap,
        "roc_auc": roc,
        "reverse_roc_auc": roc_reverse,
    })

result_df = pd.DataFrame(results)

print(
    result_df.sort_values(
        "average_precision",
        ascending=False
    ).to_string(index=False)
)

print("\n========== FEATURE DISTRIBUTION BY TARGET ==========")

for feature in FEATURES:

    print(f"\nFeature: {feature}")

    print(
        df.groupby("isFraud", observed=True)[feature]
        .describe()
        .round(4)
    )

print("\n========== EXACT TARGET MATCH CHECK ==========")

for feature in FEATURES:

    if feature == "type":
        continue

    values = df[feature]

    if values.nunique() <= 20:

        match = (
            values.astype(str)
            == y.astype(str)
        ).mean()

        print(
            f"{feature}: "
            f"{match:.6%} exact match with target"
        )

print("\n========== AUDIT COMPLETE ==========")

