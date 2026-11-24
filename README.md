# SentinelPay-AI: Real-Time Fraud Detection & Microservice Architecture

SentinelPay-AI is an end-to-end, leak-free machine learning framework and low-latency microservice architecture designed for real-time mobile money and transaction fraud detection. By combining behavioral transaction diagnostics, leak-free chronological splitting, tree ensemble gradient boosting, and explainable AI (SHAP), SentinelPay-AI minimizes false positives while detecting sophisticated fraud vectors.

---

## 📌 Architecture & System Flow

```
[ Frontend / Client Apps ]
            │
            ▼
┌──────────────────────────────────────┐
│  Java Spring Boot Enterprise Backend │ (Transaction Orchestration & Ledger)
└──────────────────┬───────────────────┘
                   │  POST /api/v1/predict (JSON payload)
                   ▼
┌──────────────────────────────────────┐
│    Python FastAPI ML Engine         │ (FastAPI + XGBoost Serving)
│  - Categorical One-Hot Encoding      │
│  - Non-linear logit inference        │  Latency: < 10ms
│  - Threshold Optimization (0.9928)   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│      Decision & Risk Policy          │ -> [ APPROVE | CHALLENGE (MFA) | BLOCK ]
└──────────────────────────────────────┘
```

---

## 🔬 Phase 1: Machine Learning Pipeline Overview

Financial fraud detection presents distinct challenges:
- **Severe Class Imbalance**: Fraud transactions represent < 0.15% of historical traffic.
- **Data Leakage Risk**: Post-transaction balances (`newbalanceOrig`, `newbalanceDest`) enable trivial algebraic leakage. In SentinelPay-AI, all post-transaction columns were permanently excised during the leakage audit (`06_feature_leakage_audit.py`).
- **Feature Robustness (Ratio vs. No-Ratio)**: Ablation studies (`14_random_forest_feature_ablation.py` and `15_xgboost_no_ratio.py`) proved that training XGBoost on direct balance and volume features without numerical ratio denominators avoids division artifacts while delivering superior ranking performance.

### Repository Structure

```
SentinelPay-AI/
├── ml-engine/                       # Phase 2 FastAPI ML microservice
│   ├── main.py                      # REST endpoints for Spring Boot integration
│   ├── predictor.py                 # Self-contained preprocessor & XGBoost predictor
│   └── requirements.txt             # Service dependencies
├── models/
│   └── baseline/
│       ├── xgboost_no_ratio.json    # Selected production XGBoost model
│       └── xgboost_no_ratio_preprocessing.json # Encoders and feature layouts
├── notebooks/
│   ├── 01_dataset_overview.py          # Data ingestion and structure
│   ├── 02_exploratory_analysis.ipynb   # Imbalance & distribution diagnostics
│   ├── 03_balance_diagnostics.py       # Account balance consistency checks
│   ├── 04_feature_engineering.py       # Pre-transaction feature creation
│   ├── 05_validate_features.py         # Null audits & assertion suites
│   ├── 06_feature_leakage_audit.py     # Leakage elimination (post-balance removal)
│   ├── 07_chronological_split.py       # 70/15/15 chronological step split
│   ├── 08_preprocessing.py             # Encoders & scalers
│   ├── 09_baseline_model_training.py   # Logistic Regression benchmark
│   ├── 10_random_forest.py             # Random Forest benchmark
│   ├── 11_random_forest_diagnostics.py # Calibration & PR curves
│   ├── 12_fraud_pattern_analysis.py    # Typology profiling
│   ├── 13_final_test_evaluation.py     # Multi-model evaluation on held-out steps
│   ├── 14_random_forest_feature_ablation.py # Ratio vs. non-ratio ablation
│   ├── 15_xgboost_no_ratio.py          # Selected XGBoost model pipeline
│   ├── 16_xgboost_shap_explainability.py # Global & local SHAP waterfall analysis
│   └── 17_temporal_drift_analysis.py   # Chronological drift & segment evaluation
└── reports/
    ├── final_ml_evaluation_report.md       # Comprehensive evaluation audit
    ├── temporal_drift_analysis.json        # Drift analysis across time segments
    ├── top_false_positive_investigation.json # SHAP investigation of top false positive
    ├── xgboost_no_ratio_metrics.json       # Candidate benchmark metrics
    ├── xgboost_shap_summary.json           # SHAP summary & top importances
    └── figures/shap/                       # Waterfall, bar, and beeswarm visualizations
```

---

## 📈 Benchmark Performance & Comparison

Models were evaluated across chronological splits (Train: Steps 1–520; Val: Steps 521–631; Test: Steps 632–743):

| Model Architecture | Features | PR-AUC (Val) | ROC-AUC (Val) | Precision (Val) | Recall (Val) | F1-Score (Val) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | Linear / Scaled | 0.3543 | 0.9676 | 0.8872 | 0.7712 | 0.8251 |
| **Random Forest** | Full + Ratios | 0.8912 | 0.9991 | 0.9502 | 0.8250 | 0.8832 |
| **XGBoost (Selected)** | **No Ratios** | **0.9575** | **0.9996** | **0.9361** | **0.8559** | **0.8942** |

### Test Performance (Steps 632–743):
- **PR-AUC:** `0.967916`
- **ROC-AUC:** `0.999407`
- **F1 Threshold:** `0.992821`
- **Precision:** `95.84%`
- **Recall:** `82.91%`
- **F1 Score:** `88.91%`

---

## 🔍 Explainability & False Positive Diagnostics

Using Tree SHAP (`notebooks/16_xgboost_shap_explainability.py` and `reports/top_false_positive_investigation.json`):

1. **Global Predictive Drivers:**
   - Pre-transaction balance (`oldbalanceOrg`) and transaction `amount` are the dominant continuous signals.
   - Modality indicators (`type_TRANSFER` and `type_CASH_OUT`) concentrate virtually all fraud activity.
2. **Top False Positive Analysis (Validation Row 62,103):**
   - A legitimate transfer of \$656,256.76 emptying an origin account balance of \$669,431.00 into a \$0-balance recipient yielded a model score of `0.999590`.
   - **Ground-Truth Context:** Among validation transactions matching this exact profile (high-value transfers liquidating >95% of an account into an empty destination), **98.86% are confirmed fraud**. The model's assessment accurately reflects extreme statistical risk; rather than overfitting to silence this point, high-probability scores are routed to step-up multi-factor authentication (MFA).

---

## ⏳ Temporal Drift Findings & Production Holdout Guidelines

Evaluating across 6 discrete time slices (Steps 521 through 743) demonstrated:
- **Ranking Stability:** PR-AUC remained between `0.9432` and `0.9856` across all segments.
- **Simulation Artifact:** Overall transaction volume drops precipitously in late PaySim steps while fraud injections remain constant, artificially elevating fraud prevalence from 0.43% to 4.67%.
- **Holdout Recommendation:** Because the test set (steps 632–743) was referenced during baseline comparisons, Phase 2 implements a **14-day continuous rolling shadow window** to validate real-time drift under live banking conditions.

---

## 🚀 Running the FastAPI Inference Service (Phase 2 Scaffolding)

### 1. Installation
```bash
cd ml-engine
python -m venv .venv
# Activate virtualenv (.venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

### 2. Start the Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Test Prediction Endpoint
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "transaction_id": "tx_12345",
       "type": "TRANSFER",
       "amount": 250000.0,
       "oldbalanceOrg": 260000.0,
       "oldbalanceDest": 0.0,
       "step": 545
     }'
```

**Response:**
```json
{
  "transaction_id": "tx_12345",
  "fraud_probability": 0.998412,
  "model_threshold": 0.992821,
  "is_above_model_threshold": true,
  "model_version": "xgboost_no_ratio_v1",
  "policy": {
    "action": "CHALLENGE",
    "challenge_threshold": 0.700000,
    "manual_review_threshold": 0.850000,
    "block_threshold": 0.999900,
    "step_up_mfa_required": true,
    "manual_review_required": true,
    "recommendation": "Route to fraud analyst queue and prompt user for Step-Up Biometric/MFA authentication."
  },
  "latency_ms": 3.842
}
```

### 4. Input Features & Derivation Specification
- `step` (int $\ge 1$): Hourly simulation timestep. When provided, automatically derives:
  - `transaction_day = (step - 1) // 24` (range: $0$ to $30$)
  - `transaction_hour = (step - 1) % 24` (range: $0$ to $23$)
- `transaction_day` (int $0 \le d \le 30$) and `transaction_hour` (int $0 \le h \le 23$): Explicit inputs override only their respective derived values.
- Pre-transaction balances `oldbalanceOrg` and `oldbalanceDest` are preserved without post-transaction leakage.
- Configurable risk policy decoupled from model score (`APPROVE` / `CHALLENGE` / `BLOCK`).

---

## 📄 License
This project is open-source under the MIT License.
