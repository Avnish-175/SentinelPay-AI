import time
from typing import Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
import uvicorn

from predictor import SentinelPayPredictor, ModelArtifactError, InputValidationError
from policy import RiskPolicyEngine, RiskPolicyConfig, RiskAction

app = FastAPI(
    title="SentinelPay-AI Inference & Policy Microservice",
    description="Real-Time Transaction Fraud Detection and Configurable Risk Decision API for Enterprise Spring Boot Integration",
    version="1.1.0",
)

# Global instances initialized on startup
predictor: Optional[SentinelPayPredictor] = None
policy_engine: Optional[RiskPolicyEngine] = None
startup_error: Optional[str] = None


@app.on_event("startup")
def startup_event():
    global predictor, policy_engine, startup_error
    try:
        predictor = SentinelPayPredictor()
        # Decouple enterprise blocking policy from model threshold:
        # Transactions above challenge_threshold (0.70) are challenged (MFA);
        # Transactions >= 0.85 undergo mandatory manual review;
        # Only transactions >= block_threshold (0.999900) are blocked outright.
        policy_config = RiskPolicyConfig(
            challenge_threshold=0.700000,
            manual_review_threshold=0.850000,
            block_threshold=0.999900,
        )
        policy_engine = RiskPolicyEngine(config=policy_config)
        startup_error = None
    except Exception as e:
        startup_error = str(e)
        predictor = None
        policy_engine = None


class RiskPolicyOverride(BaseModel):
    challenge_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.65)
    manual_review_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.80)
    block_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, example=0.999)


class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = Field(None, example="tx_987654321")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] = Field(..., example="TRANSFER")
    amount: float = Field(..., ge=0.0, example=150000.0)
    oldbalanceOrg: float = Field(..., ge=0.0, example=150000.0)
    oldbalanceDest: float = Field(..., ge=0.0, example=0.0)
    step: Optional[int] = Field(None, ge=1, example=545)
    transaction_day: Optional[int] = Field(None, ge=0, le=30, example=22)
    transaction_hour: Optional[int] = Field(None, ge=0, le=23, example=16)
    policy_override: Optional[RiskPolicyOverride] = None

    @field_validator("amount", "oldbalanceOrg", "oldbalanceDest")
    @classmethod
    def validate_finite(cls, v: float, info) -> float:
        import math
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"{info.field_name} must be finite and not NaN")
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative")
        return v


class PolicyEvaluation(BaseModel):
    action: RiskAction = Field(..., example="CHALLENGE")
    challenge_threshold: float
    manual_review_threshold: Optional[float]
    block_threshold: float
    step_up_mfa_required: bool
    manual_review_required: bool
    recommendation: str


class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float
    model_threshold: float
    is_above_model_threshold: bool
    model_version: str
    policy: PolicyEvaluation
    latency_ms: float


@app.exception_handler(InputValidationError)
async def handle_input_validation_error(request: Request, exc: InputValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Input validation error", "error": str(exc)},
    )


@app.exception_handler(ModelArtifactError)
async def handle_model_artifact_error(request: Request, exc: ModelArtifactError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Model artifact error", "error": str(exc)},
    )


@app.get("/health")
def health_check():
    if startup_error is not None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "UNHEALTHY",
                "ready": False,
                "error": startup_error,
            },
        )

    if predictor is None or policy_engine is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "INITIALIZING",
                "ready": False,
                "error": "Engine components not initialized",
            },
        )

    return {
        "status": "HEALTHY",
        "ready": True,
        "model_version": predictor.model_version,
        "model_threshold": predictor.decision_threshold,
        "features_loaded": len(predictor.expected_feature_names),
        "policy_engine_active": True,
        "default_challenge_threshold": policy_engine.config.challenge_threshold,
        "default_manual_review_threshold": policy_engine.config.manual_review_threshold,
        "default_block_threshold": policy_engine.config.block_threshold,
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
def evaluate_transaction(tx: TransactionRequest):
    start_time = time.perf_counter()

    if predictor is None or policy_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference engine unavailable: {startup_error or 'Startup incomplete'}",
        )

    try:
        # Step 1: Pure ML Model Scoring (Model threshold & score untouched)
        ml_result = predictor.predict_proba(
            transaction_type=tx.type,
            amount=tx.amount,
            oldbalance_org=tx.oldbalanceOrg,
            oldbalance_dest=tx.oldbalanceDest,
            step=tx.step,
            transaction_day=tx.transaction_day,
            transaction_hour=tx.transaction_hour,
        )

        # Step 2: Enterprise Risk Policy Evaluation (Separated from model)
        active_policy = policy_engine
        if tx.policy_override is not None:
            c_thresh = (
                tx.policy_override.challenge_threshold
                if tx.policy_override.challenge_threshold is not None
                else policy_engine.config.challenge_threshold
            )
            m_thresh = (
                tx.policy_override.manual_review_threshold
                if tx.policy_override.manual_review_threshold is not None
                else policy_engine.config.manual_review_threshold
            )
            b_thresh = (
                tx.policy_override.block_threshold
                if tx.policy_override.block_threshold is not None
                else policy_engine.config.block_threshold
            )
            try:
                override_cfg = RiskPolicyConfig(
                    challenge_threshold=c_thresh,
                    manual_review_threshold=m_thresh,
                    block_threshold=b_thresh,
                )
                active_policy = RiskPolicyEngine(config=override_cfg)
            except ValueError as cfg_err:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid policy override: {cfg_err}",
                )

        policy_res = active_policy.evaluate(
            fraud_probability=ml_result["fraud_probability"],
            model_threshold=ml_result["model_threshold"],
            transaction_amount=tx.amount,
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)

        return PredictionResponse(
            transaction_id=tx.transaction_id,
            fraud_probability=ml_result["fraud_probability"],
            model_threshold=ml_result["model_threshold"],
            is_above_model_threshold=ml_result["is_above_model_threshold"],
            model_version=ml_result["model_version"],
            policy=PolicyEvaluation(
                action=policy_res["action"],
                challenge_threshold=policy_res["challenge_threshold"],
                manual_review_threshold=policy_res["manual_review_threshold"],
                block_threshold=policy_res["block_threshold"],
                step_up_mfa_required=policy_res["step_up_mfa_required"],
                manual_review_required=policy_res["manual_review_required"],
                recommendation=policy_res["recommendation"],
            ),
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except (InputValidationError, ValueError) as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as runtime_err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution failure: {runtime_err}",
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
