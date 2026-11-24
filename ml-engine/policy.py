from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any


RiskAction = Literal["APPROVE", "CHALLENGE", "BLOCK"]


@dataclass
class RiskPolicyConfig:
    """
    Configurable risk thresholds for decision routing.
    
    Attributes:
        challenge_threshold: Minimum fraud probability to trigger step-up verification (e.g. OTP/SMS).
        manual_review_threshold: Minimum fraud probability within CHALLENGE to require fraud analyst review alongside MFA.
        block_threshold: Minimum fraud probability to immediately reject/block transaction unconditionally.
    """
    challenge_threshold: float = 0.700000
    manual_review_threshold: Optional[float] = 0.850000
    block_threshold: float = 0.999900  # Decoupled from ML threshold (0.992821); blocks only near-certain attacks

    def __post_init__(self):
        if not (0.0 <= self.challenge_threshold <= 1.0):
            raise ValueError(f"challenge_threshold must be between 0 and 1, got {self.challenge_threshold}")
        if not (0.0 <= self.block_threshold <= 1.0):
            raise ValueError(f"block_threshold must be between 0 and 1, got {self.block_threshold}")
        if self.manual_review_threshold is not None:
            if not (0.0 <= self.manual_review_threshold <= 1.0):
                raise ValueError(f"manual_review_threshold must be between 0 and 1, got {self.manual_review_threshold}")
            if self.manual_review_threshold < self.challenge_threshold:
                raise ValueError(
                    f"manual_review_threshold ({self.manual_review_threshold}) cannot be less than challenge_threshold ({self.challenge_threshold})"
                )
            if self.manual_review_threshold > self.block_threshold:
                raise ValueError(
                    f"manual_review_threshold ({self.manual_review_threshold}) cannot exceed block_threshold ({self.block_threshold})"
                )
        if self.challenge_threshold > self.block_threshold:
            raise ValueError(
                f"challenge_threshold ({self.challenge_threshold}) cannot exceed block_threshold ({self.block_threshold})"
            )


class RiskPolicyEngine:
    """
    Separated policy decision engine. Evaluates statistical ML model probability
    against enterprise governance policies, allowing differentiated handling
    such as Step-Up MFA, manual investigation queues, or outright blocking.
    """

    def __init__(self, config: Optional[RiskPolicyConfig] = None):
        self.config = config or RiskPolicyConfig()

    def evaluate(
        self,
        fraud_probability: float,
        model_threshold: float,
        transaction_amount: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate risk policy for a transaction prediction.
        """
        if not (0.0 <= fraud_probability <= 1.0):
            raise ValueError(f"Invalid fraud_probability: {fraud_probability}")

        # Evaluate risk tiers
        if fraud_probability >= self.config.block_threshold:
            action: RiskAction = "BLOCK"
            recommendation = "Reject transaction due to critical risk score matching known drain vectors."
            review_required = False
            mfa_step_up = False
        elif fraud_probability >= self.config.challenge_threshold:
            action = "CHALLENGE"
            mfa_step_up = True
            if self.config.manual_review_threshold and fraud_probability >= self.config.manual_review_threshold:
                review_required = True
                recommendation = "Route to fraud analyst queue and prompt user for Step-Up Biometric/MFA authentication."
            else:
                review_required = False
                recommendation = "Challenge transaction with SMS/OTP or Step-Up Push Authentication."
        else:
            action = "APPROVE"
            recommendation = "Approve transaction without friction."
            review_required = False
            mfa_step_up = False

        return {
            "action": action,
            "policy_rule_triggered": f"probability_band_{action.lower()}",
            "challenge_threshold": self.config.challenge_threshold,
            "manual_review_threshold": self.config.manual_review_threshold,
            "block_threshold": self.config.block_threshold,
            "step_up_mfa_required": mfa_step_up,
            "manual_review_required": review_required,
            "recommendation": recommendation,
        }
