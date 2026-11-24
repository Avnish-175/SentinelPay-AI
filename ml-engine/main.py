from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import uvicorn
from predictor import SentinelPayPredictor

app = FastAPI(
    title="SentinelPay-AI Inference Engine",
    description="Real-Time Transaction Fraud Detection API for Java Spring Boot Integration",
    version="1.0.0"
)

predictor = None

@app.on_event("startup")
def load_engine():
    global predictor
    predictor = SentinelPayPredictor()

class TransactionRequest(BaseModel):
    transaction_id: Optional[str] = Field(None, example="tx_987654321")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] = Field(..., example="TRANSFER")
    amount: float = Field(..., ge=0.0, example=150000.0)
    oldbalanceOrg: float = Field(..., ge=0.0, example=150000.0)
    oldbalanceDest: float = Field(..., ge=0.0, example=0.0)
    step: Optional[int] = Field(None, ge=1, example=545)
    transaction_day: Optional[int] = Field(None, ge=1, le=31, example=22)
    transaction_hour: Optional[int] = Field(None, ge=0, le=23, example=16)

class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float
    decision_threshold: float
    is_fraud: bool
    action: Literal["APPROVE", "CHALLENGE", "BLOCK"]
    model_version: str = "xgboost_no_ratio_v1"

@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "model_loaded": predictor is not None}

@app.post("/api/v1/predict", response_model=PredictionResponse)
def evaluate_transaction(tx: TransactionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model engine initializing")
    
    res = predictor.predict(
        transaction_type=tx.type,
        amount=tx.amount,
        oldbalance_org=tx.oldbalanceOrg,
        oldbalance_dest=tx.oldbalanceDest,
        step=tx.step,
        transaction_day=tx.transaction_day,
        transaction_hour=tx.transaction_hour,
    )

    return PredictionResponse(
        transaction_id=tx.transaction_id,
        fraud_probability=res["fraud_probability"],
        decision_threshold=res["decision_threshold"],
        is_fraud=res["is_fraud"],
        action=res["action"],
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
