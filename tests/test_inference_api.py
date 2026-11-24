import sys
from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "ml-engine"))

from predictor import (
    SentinelPayPredictor,
    ModelArtifactError,
    InputValidationError,
    ALLOWED_TYPES,
)
from policy import RiskPolicyEngine, RiskPolicyConfig
from main import app, startup_event


@pytest.fixture(scope="session")
def predictor_instance():
    return SentinelPayPredictor()


@pytest.fixture(scope="session")
def test_client():
    startup_event()
    return TestClient(app)


# ============================================================
# 1. RISK POLICY SEPARATION & BAND TESTS
# ============================================================

def test_risk_policy_bands():
    cfg = RiskPolicyConfig(
        challenge_threshold=0.70,
        manual_review_threshold=0.85,
        block_threshold=0.9999,
    )
    policy = RiskPolicyEngine(cfg)

    # Approve band (< 0.70)
    res_app = policy.evaluate(0.15, 0.992821, 500.0)
    assert res_app["action"] == "APPROVE"
    assert not res_app["step_up_mfa_required"]
    assert not res_app["manual_review_required"]

    # Challenge band with OTP/MFA (0.70 <= prob < 0.85)
    res_chal1 = policy.evaluate(0.75, 0.992821, 10000.0)
    assert res_chal1["action"] == "CHALLENGE"
    assert res_chal1["step_up_mfa_required"] is True
    assert res_chal1["manual_review_required"] is False

    # Challenge band with manual review and MFA (0.85 <= prob < 0.9999)
    res_chal2 = policy.evaluate(0.95, 0.992821, 10000.0)
    assert res_chal2["action"] == "CHALLENGE"
    assert res_chal2["step_up_mfa_required"] is True
    assert res_chal2["manual_review_required"] is True

    # Block band (>= 0.9999)
    res_blk = policy.evaluate(0.99995, 0.992821, 500000.0)
    assert res_blk["action"] == "BLOCK"
    assert not res_blk["step_up_mfa_required"]
    assert not res_blk["manual_review_required"]


def test_invalid_policy_threshold_validations():
    # challenge_threshold > block_threshold
    with pytest.raises(ValueError, match="cannot exceed block_threshold"):
        RiskPolicyConfig(challenge_threshold=0.95, block_threshold=0.70)

    # negative challenge_threshold
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        RiskPolicyConfig(challenge_threshold=-0.1)

    # block_threshold > 1
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        RiskPolicyConfig(block_threshold=1.05)

    # manual_review_threshold < challenge_threshold
    with pytest.raises(ValueError, match="cannot be less than challenge_threshold"):
        RiskPolicyConfig(challenge_threshold=0.70, manual_review_threshold=0.60, block_threshold=0.99)

    # manual_review_threshold > block_threshold
    with pytest.raises(ValueError, match="cannot exceed block_threshold"):
        RiskPolicyConfig(challenge_threshold=0.70, manual_review_threshold=0.999, block_threshold=0.95)


# ============================================================
# 2. FEATURE PARITY & DAY/HOUR INDEPENDENT OVERRIDES
# ============================================================

def test_day_hour_derivation_parity(predictor_instance):
    # step=545 -> (545 - 1) // 24 = 22, (545 - 1) % 24 = 16
    csr_derived = predictor_instance.transform("TRANSFER", 1000.0, 1000.0, 0.0, step=545)
    
    # Explicit matching day=22, hour=16 without step
    csr_explicit = predictor_instance.transform("TRANSFER", 1000.0, 1000.0, 0.0, transaction_day=22, transaction_hour=16)
    np.testing.assert_allclose(csr_derived.toarray(), csr_explicit.toarray())

    # Partial override: step=545 (day=22, hour=16) with explicit transaction_hour=10
    csr_partial = predictor_instance.transform("TRANSFER", 1000.0, 1000.0, 0.0, step=545, transaction_hour=10)
    # Check that day is 22 and hour is 10
    features = predictor_instance.expected_feature_names
    day_idx = features.index("transaction_day")
    hour_idx = features.index("transaction_hour")
    assert csr_partial[0, day_idx] == 22.0
    assert csr_partial[0, hour_idx] == 10.0


def test_transaction_day_range_validation(predictor_instance):
    # Valid range is 0 to 30
    predictor_instance.transform("PAYMENT", 10.0, 100.0, 0.0, transaction_day=0)
    predictor_instance.transform("PAYMENT", 10.0, 100.0, 0.0, transaction_day=30)

    with pytest.raises(InputValidationError):
        predictor_instance.transform("PAYMENT", 10.0, 100.0, 0.0, transaction_day=-1)

    with pytest.raises(InputValidationError):
        predictor_instance.transform("PAYMENT", 10.0, 100.0, 0.0, transaction_day=31)


# ============================================================
# 3. FALSE POSITIVE REGRESSION TEST (CRITICAL TEST)
# ============================================================

def test_known_false_positive_regression_predictor(predictor_instance):
    """
    Validates that the known highest-scored legitimate validation transaction
    is scored with high probability (~0.999590) by the untouched ML model,
    exceeds the ML validation threshold (0.992821), yet remains distinct from policy.
    """
    res = predictor_instance.predict_proba(
        transaction_type="TRANSFER",
        amount=656256.76,
        oldbalance_org=669431.0,
        oldbalance_dest=0.0,
        step=545,
    )
    expected_prob = 0.9995900988578796
    assert np.isclose(res["fraud_probability"], expected_prob, rtol=1e-5, atol=1e-6)
    assert res["is_above_model_threshold"] is True
    assert np.isclose(res["model_threshold"], 0.9928211569786072, rtol=1e-5)


def test_known_false_positive_policy_decision(test_client):
    """
    REGRESSION REQUIREMENT:
    Under the default risk policy, the known false positive transaction:
    - Amount: 656256.76
    - OldbalanceOrg: 669431.0
    - OldbalanceDest: 0.0
    - Step: 545
    Must receive:
      - action: CHALLENGE
      - step_up_mfa_required: True
      - manual_review_required: True
    and must NOT be automatically blocked solely because it exceeds the ML threshold.
    """
    payload = {
        "transaction_id": "tx_known_fp_validation_row62103",
        "type": "TRANSFER",
        "amount": 656256.76,
        "oldbalanceOrg": 669431.0,
        "oldbalanceDest": 0.0,
        "step": 545,
    }
    response = test_client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "tx_known_fp_validation_row62103"
    assert np.isclose(data["fraud_probability"], 0.999590, atol=1e-4)
    assert data["is_above_model_threshold"] is True

    # Policy verification
    policy = data["policy"]
    assert policy["action"] == "CHALLENGE"
    assert policy["step_up_mfa_required"] is True
    assert policy["manual_review_required"] is True
    assert "fraud analyst" in policy["recommendation"].lower() or "step-up" in policy["recommendation"].lower()


# ============================================================
# 4. FASTAPI ENDPOINT TESTS & OVERRIDES
# ============================================================

def test_api_health(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["ready"] is True
    assert "model_version" in data
    assert np.isclose(data["model_threshold"], 0.992821, atol=1e-4)
    assert data["default_block_threshold"] == 0.9999
    assert data["default_challenge_threshold"] == 0.70
    assert data["default_manual_review_threshold"] == 0.85
    assert data["features_loaded"] == 14


def test_api_predict_approve(test_client):
    payload = {
        "transaction_id": "tx_test_app_01",
        "type": "PAYMENT",
        "amount": 25.50,
        "oldbalanceOrg": 500.0,
        "oldbalanceDest": 200.0,
        "step": 150,
    }
    response = test_client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["policy"]["action"] == "APPROVE"
    assert not data["policy"]["step_up_mfa_required"]
    assert not data["policy"]["manual_review_required"]
    assert data["fraud_probability"] < 0.10
    assert data["latency_ms"] >= 0.0


def test_api_predict_block_via_configurable_override(test_client):
    """
    If an institution wishes to enforce a strict block threshold (e.g. 0.990)
    for high-risk transfers, verify the policy override functions correctly.
    """
    payload = {
        "transaction_id": "tx_override_block",
        "type": "TRANSFER",
        "amount": 656256.76,
        "oldbalanceOrg": 669431.0,
        "oldbalanceDest": 0.0,
        "step": 545,
        "policy_override": {
            "block_threshold": 0.990000,
        },
    }
    response = test_client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["policy"]["action"] == "BLOCK"
    assert data["policy"]["block_threshold"] == 0.990000


def test_api_invalid_policy_override_rejection(test_client):
    payload = {
        "type": "TRANSFER",
        "amount": 100.0,
        "oldbalanceOrg": 100.0,
        "oldbalanceDest": 0.0,
        "policy_override": {
            "challenge_threshold": 0.95,
            "block_threshold": 0.60,  # Invalid: challenge > block
        },
    }
    response = test_client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    assert "Invalid policy override" in response.json()["detail"]


def test_api_predict_malformed_requests(test_client):
    # Missing required field 'amount'
    res1 = test_client.post("/api/v1/predict", json={"type": "TRANSFER", "oldbalanceOrg": 100.0, "oldbalanceDest": 0.0})
    assert res1.status_code == 422

    # Negative amount
    res2 = test_client.post("/api/v1/predict", json={"type": "TRANSFER", "amount": -50.0, "oldbalanceOrg": 100.0, "oldbalanceDest": 0.0})
    assert res2.status_code == 422

    # Invalid type
    res3 = test_client.post("/api/v1/predict", json={"type": "BITCOIN", "amount": 50.0, "oldbalanceOrg": 100.0, "oldbalanceDest": 0.0})
    assert res3.status_code == 422
