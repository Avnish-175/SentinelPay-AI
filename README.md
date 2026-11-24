# SentinelPay-AI: Real-Time & High-Precision Fraud Detection System

SentinelPay-AI is an end-to-end machine learning framework designed for real-time mobile money and financial transaction fraud detection. By combining behavioral transaction diagnostics, leak-free chronological splitting, domain-specific feature engineering, and robust tree ensembles, SentinelPay-AI minimizes false positives while capturing fraud patterns.

---

## 📌 Project Overview

Financial fraud detection presents significant machine learning challenges:
- **Severe Class Imbalance**: Fraudulent transactions account for a minuscule fraction of all transfers (< 0.15%).
- **Temporal Drift**: Transaction patterns change over time, rendering naive cross-validation prone to data leakage.
- **Account Discrepancies**: Subtle mismatches between initial balances, transaction amounts, and ending balances often indicate fraudulent drains.

SentinelPay-AI addresses these challenges through a disciplined pipeline from raw data diagnostics to calibrated model deployment.

---

## 📊 Pipeline Architecture & Workflow

The repository is organized sequentially across data auditing, feature engineering, and modeling stages:

```
SentinelPay-AI/
├── data/
│   ├── artifacts/preprocessing/     # Saved feature names & preprocessor objects
│   └── processed/                   # Aggregated statistics and diagnostic plots
├── models/
│   └── baseline/                    # Serialized baseline models (Logistic Regression, RF)
├── notebooks/
│   ├── 01_dataset_overview.py          # Data ingestion and structural summary
│   ├── 02_exploratory_analysis.ipynb   # Imbalance, transaction volume & class analysis
│   ├── 03_balance_diagnostics.py       # Origin/dest account balance integrity checks
│   ├── 04_feature_engineering.py       # Behavioral ratios, temporal tags & flag indicators
│   ├── 05_validate_features.py         # Null audits, zero-variance checks & assertions
│   ├── 06_feature_leakage_audit.py     # Target leakage and time-travel verification
│   ├── 07_chronological_split.py       # Leakage-free train/val/test temporal splitting
│   ├── 08_preprocessing.py             # Encoders, scalers & tree/linear pipelines
│   ├── 09_baseline_model_training.py   # Logistic Regression benchmark
│   ├── 10_random_forest.py             # Subsampled balanced Random Forest classifier
│   ├── 11_random_forest_diagnostics.py # Calibration curve, PR curves & threshold tuning
│   └── 12_fraud_pattern_analysis.py    # False positive/negative profiling & rules
└── reports/
    ├── logistic_regression_metrics.json    # LR performance benchmark
    ├── random_forest_metrics.json          # Random Forest metrics across thresholds
    └── random_forest_feature_importance.csv# Top feature importances
```

---

## 🔬 Key Engineering Highlights

1. **Chronological Splitting**: Rather than random K-Fold splits, SentinelPay-AI employs strict time-ordered splits (`step`-based) to simulate real production deployment and prevent lookahead bias.
2. **Behavioral Feature Engineering**:
   - `amount_to_origin_balance_ratio`: Detects full account liquidations.
   - `amount_to_destination_balance_ratio`: Identifies sudden massive influxes into dormant or newly created accounts.
   - Temporal decomposition: Cyclical transaction hour extraction.
   - Balance error flags: Discrepancy checks identifying balance falsification.
3. **Threshold Optimization & Diagnostics**:
   - Beyond raw ROC-AUC, evaluation is anchored on **Precision-Recall AUC (PR-AUC)**, cost-sensitive matrices, and optimal F1 thresholds.

---

## 📈 Benchmark Performance

Evaluated on held-out temporal test splits:

| Model | ROC-AUC | PR-AUC | Optimal Threshold | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.9676 | 0.3543 | 0.9997 | 0.8872 | 0.7712 | 0.8251 |
| **Balanced Random Forest** | **1.0000** | **1.0000** | 0.3675 | **1.0000** | **1.0000** | **1.0000** |

### Top Predictive Features:
1. `amount_to_origin_balance_ratio` (~41.3% importance)
2. `oldbalanceOrg` (~14.9% importance)
3. `amount` (~11.6% importance)
4. `type_PAYMENT` / `type_TRANSFER`

---

## 🚀 Getting Started

### 1. Prerequisites & Environment
Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/Avnish-175/SentinelPay-AI.git
cd SentinelPay-AI
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn joblib matplotlib seaborn
```

### 2. Dataset Setup
Download the [Paysim Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) and place `PS_20174392719_1491204439457_log.csv` into:
```
data/raw/PS_20174392719_1491204439457_log.csv
```

### 3. Running the Pipeline
Execute the pipeline scripts sequentially from the `notebooks/` directory:

```bash
python notebooks/04_feature_engineering.py
python notebooks/07_chronological_split.py
python notebooks/08_preprocessing.py
python notebooks/10_random_forest.py
```

---

## 📄 License
This project is open-source under the MIT License.
