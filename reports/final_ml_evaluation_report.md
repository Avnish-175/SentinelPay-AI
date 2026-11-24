# SentinelPay-AI: Phase 1 Final Machine Learning Evaluation Report

## 1. Executive Summary & Objective

**SentinelPay-AI** is an advanced transaction fraud monitoring system developed to identify illicit financial flows in high-volume, real-time payment environments. 

Phase 1 focused on building a leak-free, empirically verified machine learning pipeline on the PaySim synthetic mobile payment dataset (6.36 million transactions), benchmarking candidate architectures, auditing feature leakage, generating explainable SHAP diagnostics, and establishing real-world operational boundaries.

### Core Benchmark Results:
- **Selected Model:** **XGBoost (without ratio features)**
- **Hyperparameter Profile:** 500 trees, `max_depth=6`, `learning_rate=0.05`, `scale_pos_weight=1051.07`, `tree_method='hist'`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Validation PR-AUC:** **0.957479**
- **Validation ROC-AUC:** **0.999634**
- **Selected Decision Threshold (F1-optimal on validation):** **0.992821**
- **Validation Precision / Recall / F1:** **93.61% / 85.59% / 89.42%**
- **Test Set Metrics (Steps 632–743):**
  - PR-AUC: **0.967916**
  - ROC-AUC: **0.999407**
  - Precision: **95.84%**
  - Recall: **82.91%**
  - F1-Score: **88.91%**

---

## 2. Feature Selection & Leakage Auditing Decisions

### 2.1 Post-Transaction Target Leakage Elimination
Initial explorations revealed that `newbalanceOrig` and `newbalanceDest` allowed trivial algebraic deduction of fraud flags in PaySim (e.g., `newbalanceOrig == 0` when `oldbalanceOrg > 0` during high-value transfers). In a real-time pre-authorization payment gateway, **post-transaction balances are unavailable prior to execution**.
- **Action:** All downstream balances (`newbalanceOrig`, `newbalanceDest`, and post-transaction discrepancy formulas) were strictly excised during the leakage audit (`06_feature_leakage_audit.py`).
- **Approved Feature Set:** 
  1. Transaction category: `type` (One-Hot Encoded: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
  2. Transaction amount: `amount`
  3. Pre-transaction balances: `oldbalanceOrg`, `oldbalanceDest`
  4. Balance indicators: `origin_is_zero`, `destination_is_zero`
  5. Temporal features: `transaction_day`, `transaction_hour`, `hour_sin`, `hour_cos`

### 2.2 Ratio Feature Ablation
Engineered ratios such as `amount_to_origin_balance_ratio = amount / (oldbalanceOrg + 1)` produced near-1.0 metric illusions in initial Random Forest models. However, feature ablation experiments revealed:
- When ratio features were fed to Random Forest models, decision trees created sharp axis splits around `ratio ~ 1.0`, obscuring multi-feature risk interactions.
- When evaluating XGBoost without ratio features, the model achieved equivalent or superior ranking ability (PR-AUC 0.9575) by learning non-linear interactions directly between `amount` and `oldbalanceOrg` in log-loss gradient space, while avoiding numerical division instability and extreme sensitivity to near-zero balances.
- **Decision:** The production candidate model is strictly standardized on the **no-ratio feature set** (`xgboost_no_ratio.json`).

---

## 3. False Positive Investigation (Task 1)

### 3.1 Case Profile: Highest Scored Legitimate Validation Transaction
In validation split evaluations, the model assigned an extreme fraud probability of **0.999590** to a legitimate transaction (`isFraud = 0`).

| Feature Attribute | Value | Domain Interpretation |
|---|---|---|
| **Validation Index** | Row 62,103 | Step 545 (Day 22, Hour 16) |
| **Transaction Type** | `TRANSFER` | High-risk modality |
| **Amount** | \$656,256.76 | High transaction volume |
| **Old Balance Origin** | \$669,431.00 | Customer balance prior to transfer |
| **Old Balance Destination** | \$0.00 | Recipient account has \$0 pre-balance |
| **Origin Is Zero** | 0 | Originator had funds |
| **Destination Is Zero**| 1 | Recipient is an empty / newly active account |
| **Ratio (Implicit)** | ~0.9803 | 98.03% of entire account drained |

### 3.2 Individual SHAP Waterfall Decomposition
The explainer base margin is `+0.314267`. Individual SHAP contributions on the logit margin scale:
- `+2.397787`: `oldbalanceOrg = 669,431.00`
- `+1.583866`: `amount = 656,256.76`
- `+1.488673`: `transaction_day = 22`
- `+1.443010`: `oldbalanceDest = 0.0`
- `+0.835871`: `type_TRANSFER = 1`
- `+0.276812`: `destination_is_zero = 1`
- `-1.047474`: `type_CASH_OUT = 0`
- **Total Margin:** `7.799187` $\rightarrow$ $\sigma(7.799187) = \mathbf{0.999590}$

### 3.3 Root Cause Analysis & Justification
1. **Synthetic Dataset Characteristics:** In the PaySim simulation, fraud agents operate almost exclusively by seizing accounts and wiring the entire balance via `TRANSFER` to empty mule destination accounts.
2. **Empirical Ground Truth Check:** Across validation data, among transfers exceeding \$500,000 sent to a \$0 balance destination that clear out over 95% of the origin account balance, **98.86% (261 out of 264 transactions) are confirmed fraud**.
3. **Engineering Decision:** The model is behaving correctly as a statistical risk estimator. This transaction mimics predatory account-takeover drain mechanics. **We do not modify or overfit the model to silence this single false positive.** In a live banking system, such a transaction should trigger an automated step-up multi-factor authentication (MFA) challenge or hold for fraud analyst clearance rather than arbitrary threshold degradation.

---

## 4. Temporal Drift & Stability Analysis (Task 2)

An analysis was conducted across 6 chronological time segments spanning steps 521 to 743.

### 4.1 Segment Breakdown & Metric Stability

| Segment | Time Steps | Total Volume | Fraud Count | Prevalence | PR-AUC | Precision | Recall | F1 Score |
|---|---|---|---|---|---|---|---|---|
| **Val Early** | 521–557 | 86,674 | 370 | 0.427% | 0.9432 | 0.9296 | 0.8568 | 0.8917 |
| **Val Mid** | 558–594 | 70,359 | 408 | 0.580% | 0.9534 | 0.9295 | 0.8407 | 0.8829 |
| **Val Late** | 595–631 | 34,114 | 402 | 1.178% | 0.9744 | 0.9485 | 0.8706 | 0.9079 |
| **Test Early**| 632–669 | 20,940 | 414 | 1.977% | 0.9681 | 0.9503 | 0.7850 | 0.8598 |
| **Test Mid**  | 670–706 | 59,527 | 418 | 0.702% | 0.9652 | 0.9337 | 0.8756 | 0.9037 |
| **Test Late** | 707–743 | 8,999  | 420 | 4.667% | 0.9856 | 0.9943 | 0.8262 | 0.9025 |

### 4.2 Key Findings on Dataset Drift
- **Transaction Density Decay:** Normal payment volume sharply tapers off towards the end of the 30-day PaySim simulation (dropping from ~86k down to ~9k transactions per 36-step interval).
- **Constant Fraud Injection Rate:** Synthetic fraud attacks were injected at an invariant rate (~410 to 420 events per window). This causes an artificial prevalence surge from 0.43% to 4.67%.
- **Ranking Invariance:** Despite density swings, PR-AUC remained consistently robust (**0.9432 – 0.9856**).
- **Recommendation for Holdout:** Because the test set (`steps 632–743`) was evaluated during model selection, it cannot serve as a blind holdout. For Phase 2 production readiness, a **continuous shadow-scoring window** (or 14-day rolling buffer) must be established in the serving gateway to detect real-time concept drift.

---

## 5. PaySim vs. Real-World Production Boundaries

| Dimension | PaySim Experimental Dataset | Real-World Financial Production Environment |
|---|---|---|
| **Fraud Typologies** | Only 2 transaction types (`TRANSFER`, `CASH_OUT`) | Card-not-present (CNP), authorized push payment (APP), SIM-swap, smishing, triangulation fraud, credit laundering. |
| **Identity & Behavioral Breadth**| Customer IDs are opaque without history | Device fingerprinting, IP geolocation velocity, biometrics, behavioral session anomalies, beneficiary relationship history. |
| **Temporal Granularity** | Integer hourly steps (`1` to `743`) | Millisecond timestamps, sub-second transaction burst clustering, weekday/weekend holiday seasonality. |
| **Concept Drift** | Static synthetic simulation | Rapid adversarial adaptation, new attack vectors, fraud-ring evasion. |

---

## 6. Phase 2 Readiness: FastAPI & Java Spring Boot Architecture

To transition into Phase 2, the inference stack is decoupled into a high-throughput microservice architecture:

```
[ Frontend / Client ]
         │
         ▼
[ Java Spring Boot Backend ]  <-- Transaction Orchestration, Ledger & Persistence
         │
         │ (HTTP REST / gRPC: POST /api/v1/predict)
         ▼
[ Python FastAPI ML Engine ]  <-- Feature transformation & XGBoost inference (<10ms)
         │
         ▼
[ Decision Engine ]           <-- Threshold >= 0.9928 -> BLOCK / CHALLENGE / APPROVE
```

All preprocessing transforms and model weights (`models/baseline/xgboost_no_ratio.json`) are self-contained and ready for real-time serving.
