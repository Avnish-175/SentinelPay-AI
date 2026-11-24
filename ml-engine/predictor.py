import json
from pathlib import Path
from typing import List, Optional, Literal
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "baseline" / "xgboost_no_ratio.json"
DEFAULT_PREP_PATH = Path(__file__).resolve().parents[1] / "models" / "baseline" / "xgboost_no_ratio_preprocessing.json"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parents[1] / "reports" / "xgboost_no_ratio_metrics.json"

CATEGORICAL_FEATURES = ["type"]
NUMERICAL_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "transaction_day",
    "transaction_hour",
    "hour_sin",
    "hour_cos",
    "origin_is_zero",
    "destination_is_zero",
]
FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

class SentinelPayPredictor:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        prep_path: Optional[Path] = None,
        metrics_path: Optional[Path] = None,
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.prep_path = prep_path or DEFAULT_PREP_PATH
        self.metrics_path = metrics_path or DEFAULT_METRICS_PATH

        self._load_preprocessor()
        self._load_model()
        self._load_threshold()

    def _load_preprocessor(self):
        with open(self.prep_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.categories = [config["categorical_categories"]]
        self.encoder = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(
                        categories=self.categories,
                        handle_unknown="ignore",
                        sparse_output=True,
                        dtype=np.float32,
                    ),
                    CATEGORICAL_FEATURES,
                ),
                ("numerical", "passthrough", NUMERICAL_FEATURES),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )
        # Dummy fit using known categories and zero numerical values to initialize transformer structure
        dummy_data = {
            "type": ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
            "amount": [0.0] * 5,
            "oldbalanceOrg": [0.0] * 5,
            "oldbalanceDest": [0.0] * 5,
            "transaction_day": [1.0] * 5,
            "transaction_hour": [0.0] * 5,
            "hour_sin": [0.0] * 5,
            "hour_cos": [1.0] * 5,
            "origin_is_zero": [1.0] * 5,
            "destination_is_zero": [1.0] * 5,
        }
        import pandas as pd
        self.encoder.fit(pd.DataFrame(dummy_data)[FEATURES])

    def _load_model(self):
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))

    def _load_threshold(self):
        if self.metrics_path.exists():
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.decision_threshold = float(data.get("selected_threshold", 0.992821))
        else:
            self.decision_threshold = 0.992821

    def preprocess_payload(
        self,
        transaction_type: str,
        amount: float,
        oldbalance_org: float,
        oldbalance_dest: float,
        step: Optional[int] = None,
        transaction_day: Optional[int] = None,
        transaction_hour: Optional[int] = None,
    ):
        import pandas as pd
        if step is not None and transaction_day is None:
            day = (step - 1) // 24 + 1
            hour = (step - 1) % 24
        else:
            day = transaction_day if transaction_day is not None else 1
            hour = transaction_hour if transaction_hour is not None else 12

        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        origin_is_zero = 1.0 if oldbalance_org == 0 else 0.0
        dest_is_zero = 1.0 if oldbalance_dest == 0 else 0.0

        row = {
            "type": [transaction_type],
            "amount": [float(amount)],
            "oldbalanceOrg": [float(oldbalance_org)],
            "oldbalanceDest": [float(oldbalance_dest)],
            "transaction_day": [float(day)],
            "transaction_hour": [float(hour)],
            "hour_sin": [float(hour_sin)],
            "hour_cos": [float(hour_cos)],
            "origin_is_zero": [float(origin_is_zero)],
            "destination_is_zero": [float(dest_is_zero)],
        }
        df = pd.DataFrame(row)[FEATURES]
        return self.encoder.transform(df).tocsr().astype(np.float32)

    def predict(
        self,
        transaction_type: str,
        amount: float,
        oldbalance_org: float,
        oldbalance_dest: float,
        step: Optional[int] = None,
        transaction_day: Optional[int] = None,
        transaction_hour: Optional[int] = None,
    ):
        X = self.preprocess_payload(
            transaction_type=transaction_type,
            amount=amount,
            oldbalance_org=oldbalance_org,
            oldbalance_dest=oldbalance_dest,
            step=step,
            transaction_day=transaction_day,
            transaction_hour=transaction_hour,
        )
        prob = float(self.model.predict_proba(X)[0, 1])
        is_fraud = bool(prob >= self.decision_threshold)
        decision = "BLOCK" if is_fraud else ("CHALLENGE" if prob >= 0.70 else "APPROVE")
        
        return {
            "fraud_probability": prob,
            "decision_threshold": self.decision_threshold,
            "is_fraud": is_fraud,
            "action": decision,
        }
