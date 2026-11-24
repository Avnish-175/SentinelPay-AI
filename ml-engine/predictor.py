import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import math
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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
ALLOWED_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


class ModelArtifactError(Exception):
    """Raised when required model artifacts or configurations are missing or invalid."""
    pass


class InputValidationError(Exception):
    """Raised when inference input payload fails numerical or type validation."""
    pass


class SentinelPayPredictor:
    """
    ML Inference Engine encapsulating feature transformation and XGBoost scoring.
    Maintains strict mathematical parity with the validated notebook pipeline.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        prep_path: Optional[Path] = None,
        metrics_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.prep_path = Path(prep_path or DEFAULT_PREP_PATH)
        self.metrics_path = Path(metrics_path or DEFAULT_METRICS_PATH)

        self._validate_artifacts_exist()
        self._load_preprocessor()
        self._load_model()
        self._load_benchmark_metadata()

    def _validate_artifacts_exist(self):
        if not self.model_path.exists():
            raise ModelArtifactError(f"Model file not found at: {self.model_path}")
        if not self.prep_path.exists():
            raise ModelArtifactError(f"Preprocessing config not found at: {self.prep_path}")

    def _load_preprocessor(self):
        try:
            with open(self.prep_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

            self.categories = [self.config["categorical_categories"]]
            self.expected_feature_names = self.config["transformed_feature_names"]
            if self.config["numerical_features"] != NUMERICAL_FEATURES:
                raise ModelArtifactError("Mismatch in numerical features between config and predictor code.")

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

            # Fit with complete categories so transformer is ready for CSR sparse production
            dummy_data = {
                "type": self.config["categorical_categories"],
                "amount": [0.0] * len(self.config["categorical_categories"]),
                "oldbalanceOrg": [0.0] * len(self.config["categorical_categories"]),
                "oldbalanceDest": [0.0] * len(self.config["categorical_categories"]),
                "transaction_day": [1.0] * len(self.config["categorical_categories"]),
                "transaction_hour": [0.0] * len(self.config["categorical_categories"]),
                "hour_sin": [0.0] * len(self.config["categorical_categories"]),
                "hour_cos": [1.0] * len(self.config["categorical_categories"]),
                "origin_is_zero": [1.0] * len(self.config["categorical_categories"]),
                "destination_is_zero": [1.0] * len(self.config["categorical_categories"]),
            }
            self.encoder.fit(pd.DataFrame(dummy_data)[FEATURES])

            # Verify produced feature names match training exactly
            generated_names = [
                n.replace("categorical__", "").replace("numerical__", "")
                for n in self.encoder.get_feature_names_out()
            ]
            if generated_names != self.expected_feature_names:
                raise ModelArtifactError(
                    f"Feature name alignment failure. Got {generated_names}, expected {self.expected_feature_names}"
                )

        except Exception as e:
            if isinstance(e, ModelArtifactError):
                raise
            raise ModelArtifactError(f"Failed to initialize preprocessor: {e}") from e

    def _load_model(self):
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(self.model_path))
        except Exception as e:
            raise ModelArtifactError(f"Failed to load XGBoost model: {e}") from e

    def _load_benchmark_metadata(self):
        self.model_version = "xgboost_no_ratio_v1"
        self.decision_threshold = 0.9928211569786072
        if self.metrics_path.exists():
            try:
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                self.decision_threshold = float(metrics.get("selected_threshold", self.decision_threshold))
                self.model_version = f"{metrics.get('model', 'XGBoost').lower()}_{metrics.get('feature_set', 'no_ratio')}"
            except Exception:
                pass

    def validate_inputs(
        self,
        transaction_type: str,
        amount: float,
        oldbalance_org: float,
        oldbalance_dest: float,
        step: Optional[int] = None,
        transaction_day: Optional[int] = None,
        transaction_hour: Optional[int] = None,
    ):
        """Strict validation of incoming payload before inference."""
        if not isinstance(transaction_type, str) or transaction_type not in ALLOWED_TYPES:
            raise InputValidationError(
                f"Invalid transaction type: '{transaction_type}'. Allowed types: {sorted(ALLOWED_TYPES)}"
            )

        for name, val in [
            ("amount", amount),
            ("oldbalanceOrg", oldbalance_org),
            ("oldbalanceDest", oldbalance_dest),
        ]:
            if val is None or not isinstance(val, (int, float)):
                raise InputValidationError(f"Field '{name}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise InputValidationError(f"Field '{name}' cannot be NaN or Infinite (got {val})")
            if val < 0:
                raise InputValidationError(f"Field '{name}' cannot be negative (got {val})")

        if step is not None:
            if not isinstance(step, int) or step < 1:
                raise InputValidationError(f"Field 'step' must be an integer >= 1 (got {step})")

        if transaction_day is not None:
            if not isinstance(transaction_day, int) or not (1 <= transaction_day <= 31):
                raise InputValidationError(f"Field 'transaction_day' must be an integer between 1 and 31 (got {transaction_day})")

        if transaction_hour is not None:
            if not isinstance(transaction_hour, int) or not (0 <= transaction_hour <= 23):
                raise InputValidationError(f"Field 'transaction_hour' must be an integer between 0 and 23 (got {transaction_hour})")

    def transform(
        self,
        transaction_type: str,
        amount: float,
        oldbalance_org: float,
        oldbalance_dest: float,
        step: Optional[int] = None,
        transaction_day: Optional[int] = None,
        transaction_hour: Optional[int] = None,
    ) -> sparse.csr_matrix:
        """Transforms a validated transaction record into a float32 CSR sparse matrix."""
        self.validate_inputs(
            transaction_type,
            amount,
            oldbalance_org,
            oldbalance_dest,
            step,
            transaction_day,
            transaction_hour,
        )

        if step is not None and transaction_day is None:
            day = (step - 1) // 24 + 1
            hour = (step - 1) % 24
        else:
            day = transaction_day if transaction_day is not None else 1
            hour = transaction_hour if transaction_hour is not None else 12

        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        origin_is_zero = 1.0 if oldbalance_org == 0.0 else 0.0
        dest_is_zero = 1.0 if oldbalance_dest == 0.0 else 0.0

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

    def predict_proba(
        self,
        transaction_type: str,
        amount: float,
        oldbalance_org: float,
        oldbalance_dest: float,
        step: Optional[int] = None,
        transaction_day: Optional[int] = None,
        transaction_hour: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes model prediction strictly returning statistical fraud likelihood
        and baseline model metadata, decoupled from enterprise policy.
        """
        X = self.transform(
            transaction_type=transaction_type,
            amount=amount,
            oldbalance_org=oldbalance_org,
            oldbalance_dest=oldbalance_dest,
            step=step,
            transaction_day=transaction_day,
            transaction_hour=transaction_hour,
        )
        probabilities = self.model.predict_proba(X)
        fraud_probability = float(probabilities[0, 1])

        return {
            "fraud_probability": fraud_probability,
            "model_threshold": self.decision_threshold,
            "is_above_model_threshold": bool(fraud_probability >= self.decision_threshold),
            "model_version": self.model_version,
            "features_used": self.expected_feature_names,
        }
