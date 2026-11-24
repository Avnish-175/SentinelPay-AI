# SentinelPay-AI: Real-Time & High-Precision Fraud Detection System

SentinelPay-AI is an end-to-end machine learning framework for mobile-money transaction fraud detection. It combines chronological evaluation, transaction feature engineering, and tree-based models.

---

## Project Overview

Fraud detection is challenging because of severe class imbalance, temporal drift, and evolving transaction behavior. SentinelPay-AI uses time-ordered train, validation, and test splits to approximate deployment on future transactions.

## Pipeline Architecture

```
SentinelPay-AI/
├── data/
│   ├── artifacts/preprocessing/
│   └── processed/
├── models/baseline/
├── notebooks/
│   ├── 04_feature_engineering.py
│   ├── 06_feature_leakage_audit.py
│   ├── 07_chronological_split.py
│   ├── 08_preprocessing.py
│   ├── 09_baseline_model_training.py
│   ├── 10_random_forest.py
│   ├── 11_random_forest_diagnostics.py
│   ├── 12_fraud_pattern_analysis.py
│   ├── 13_final_test_evaluation.py
│   └── 14_random_forest_feature_ablation.py
└── reports/
```

## Engineering Highlights

1. **Chronological evaluation:** Train, validation, and test sets are separated by transaction step. Thresholds are selected on validation and held fixed for final test evaluation.
2. **Pre-transaction features:** The current model feature set uses transaction type, amount, origin/destination starting balances, and temporal features. Post-transaction balances and balance-change fields are excluded from model inputs.
3. **Leakage and feature ablation checks:** Candidate features are audited and important engineered features are removed in controlled experiments to measure their effect.
4. **Imbalanced classification:** PR-AUC, precision, recall, F1, and confusion matrices are reported alongside ROC-AUC.

## Benchmark Performance

The following results are from the chronological test split (89,466 transactions; 1,252 fraud cases; fraud prevalence 1.3994%). Classification thresholds were selected on the validation split and were not tuned on test.

| Model / feature set | ROC-AUC | PR-AUC | Fixed threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.960261 | 0.447946 | 0.914918 | 0.518929 | 0.448882 | 0.481370 |
| Random Forest, both ratios included | 1.000000 | 0.999997 | 0.367459 | 1.000000 | 0.998403 | 0.999201 |
| Random Forest, without origin ratio | 0.995031 | 0.876126 | 0.777307 | 0.925177 | 0.730831 | 0.816600 |
| Random Forest, without either ratio | 0.995096 | 0.882607 | 0.693158 | 0.885482 | 0.784345 | 0.831851 |

### Interpretation and limitations

The full-feature Random Forest achieves near-perfect performance on this PaySim test split, but the ablation results show that the origin-balance ratio contributes substantially to that score. Removing both ratio features reduces PR-AUC to 0.882607 while ROC-AUC remains 0.995096. This indicates that the model retains strong ranking ability without the ratios, while also showing that the original near-perfect result is strongly dependent on a PaySim-specific signal.

The ratio features are calculated from pre-transaction values and are not automatically temporal leakage. However, high performance on a synthetic dataset is not evidence of equivalent performance on real payment traffic. The no-ratio result is included as a conservative robustness comparison, not as proof of production readiness.

## Getting Started

### 1. Environment

```bash
git clone https://github.com/Avnish-175/SentinelPay-AI.git
cd SentinelPay-AI
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install numpy pandas scikit-learn joblib matplotlib seaborn
```

### 2. Dataset

Download the PaySim Synthetic Financial Datasets for Fraud Detection dataset and place the CSV at:

```
data/raw/PS_20174392719_1491204439457_log.csv
```

### 3. Run the pipeline

Run from the repository root:

```bash
python notebooks/04_feature_engineering.py
python notebooks/07_chronological_split.py
python notebooks/08_preprocessing.py
python notebooks/09_baseline_model_training.py
python notebooks/10_random_forest.py
python notebooks/13_final_test_evaluation.py
python notebooks/14_random_forest_feature_ablation.py
```

The raw dataset and generated CSV/model artifacts are not committed; create them locally by following the steps above.

## License

This project is open-source under the MIT License.
