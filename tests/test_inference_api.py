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
# 1. RISK POLICY SEPARATION TESTS
# ============================================================

def test_risk_policy_bands():
    cfg = RiskPolicyConfig(challenge_threshold=0.70, block_threshold=0.9928, manual_review_threshold=0.85)
    policy = RiskPolicyEngine(cfg)

    # Approve band
    res_app = policy.evaluate(0.15, 0.9928, 500.0)
    assert res_app["action"] == "APPROVE"
    assert not res_app["step_up_mfa_required"]
    assert not res_app["manual_review_required"]

    # Challenge band (Step-up MFA only)
    res_chal1 = policy.evaluate(0.75, 0.9928, 10000.0)
    assert res_chal1["action"] == "CHALLENGE"
    assert res_chal1["step_up_mfa_required"]
    assert not res_chal1["manual_review_required"]

    # Challenge band with manual review (>=0.85)
    res_chal2 = policy.evaluate(0.90, 0.9928, 10000.0)
    assert res_chal2["action"] == "CHALLENGE"
    assert res_chal2["step_up_mfa_required"]
    assert res_chal2["manual_review_required"]

    # Block band
    res_blk = policy.evaluate(0.9995, 0.9928, 500000.0)
    assert res_blk["action"] == "BLOCK"


def test_invalid_policy_config():
    with pytest.raises(ValueError):
        RiskPolicyConfig(challenge_threshold=0.95, block_threshold=0.70)
    with pytest.raises(ValueError):
        RiskPolicyConfig(challenge_threshold=-0.1)


# ============================================================
# 2. CSR SPARSE INFERENCE AND BENCHMARK CONSISTENCY
# ============================================================

def test_validation_consistency_top_fp(predictor_instance):
    """
    Verifies that the CSR sparse inference output in SentinelPayPredictor
    reproduces the exact top false positive probability 0.9995900988 from
    reports/xgboost_shap_summary.json and investigate_top_fp.py.
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


def test_validation_consistency_legitimate_payment(predictor_instance):
    res = predictor_instance.predict_proba(
        transaction_type="PAYMENT",
        amount=50.0,
        oldbalance_org=1000.0,
        oldbalance_dest=500.0,
        step=100,
    )
    assert res["fraud_probability"] < 0.01
    assert res["is_above_model_threshold"] is False


def test_input_validation_errors(predictor_instance):
    # Invalid transaction type
    with pytest.raises(InputValidationError):
        predictor_instance.predict_proba("INVALID_TYPE", 100.0, 100.0, 0.0)

    # Negative amount
    with pytest.raises(InputValidationError):
        predictor_instance.predict_proba("TRANSFER", -10.0, 100.0, 0.0)

    # NaN / Inf values
    with pytest.raises(InputValidationError):
        predictor_instance.predict_proba("TRANSFER", float("nan"), 100.0, 0.0)
    with pytest.raises(InputValidationError):
        predictor_instance.predict_proba("TRANSFER", 100.0, float("inf"), 0.0)

    # Invalid step
    with pytest.raises(InputValidationError):
        predictor_instance.predict_proba("TRANSFER", 100.0, 100.0, 0.0, step=0)


def test_missing_model_artifacts():
    with pytest.raises(ModelArtifactError):
        SentinelPayPredictor(model_path=Path("non_existent_model.json"))


# ============================================================
# 3. FASTAPI ENDPOINT AUTOMATED TESTS
# ============================================================

def test_api_health(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["ready"] is True
    assert "model_version" in data
    assert "model_threshold" in data
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
    assert data["transaction_id"] == "tx_test_app_01"
    assert data["policy"]["action"] == "APPROVE"
    assert not data["policy"]["step_up_mfa_required"]
    assert data["fraud_probability"] < 0.10
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0.0


def test_api_predict_challenge_and_override(test_client):
    # A transfer with moderate probability challenged by policy override
    payload = {
        "transaction_id": "tx_test_chal_01",
        "type": "TRANSFER",
        "amount": 656256.76,
        "oldbalanceOrg": 669431.0,
        "oldbalanceDest": 0.0,
        "step": 545,
        "policy_override": {
            "challenge_threshold": 0.50,
            "block_threshold": 0.9999,  # Higher than 0.999590 so it results in CHALLENGE
        },
    }
    response = test_client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["policy"]["action"] == "CHALLENGE"
    assert data["policy"]["step_up_mfa_required"] is True
    assert data["policy"]["block_threshold"] == 0.9999


def test_api_predict_block(test_client):
    payload = {
        "transaction_id": "tx_test_block_01",
        "type": "TRANSFER",
        "amount": 656256.76,
        "oldbalanceOrg": 669431.0,
        "oldbalanceDest": 0.0,
        "step": 545,
    }
    response = test_client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["policy"]["action"] == "BLOCK"
    assert data["is_above_model_threshold"] is True


def test_api_predict_malformed_requests(test_client):
    # Missing required field 'amount'
    res1 = test_client.post("/api/v1/predict", json={"type": "TRANSFER", "oldbalanceOrg": 100.0, "oldbalanceDest": 0.0})
    assert res1.status_code == 422

    # Negative amount
    res2 = test_client.post("/api/v1/predict", json={"type": "TRANSFER", "amount": -50.0, "oldbalanceOrg": 100.0, "oldbalanceDest": 0.0})
    assert res2.status_code == 422

    # Invalid type
    res3 = test_client.post("/api/v1/predict", json={"type": "WIRE", "amount": 50.0, "oldbalanceOrg": 100.0, "oldbalanceDest": 0.0})
    assert res3.status_code == 422
