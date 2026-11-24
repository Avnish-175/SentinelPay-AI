from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any


RiskAction = Literal["APPROVE", "CHALLENGE", "BLOCK"]


@dataclass
class RiskPolicyConfig:
    """
    Configurable risk thresholds for decision routing.
    
    Attributes:
        challenge_threshold: Minimum fraud probability to trigger step-up verification / manual review.
        block_threshold: Minimum fraud probability to immediately reject/block transaction.
        mfa_required_actions: Set of actions that mandate multi-factor step up.
    """
    challenge_threshold: float = 0.700000
    block_threshold: float = 0.992821  # Default aligns with ML validation F1-optimal threshold
    manual_review_threshold: Optional[float] = 0.850000

    def __post_init__(self):
        if not (0.0 <= self.challenge_threshold <= 1.0):
            raise ValueError(f"challenge_threshold must be between 0 and 1, got {self.challenge_threshold}")
        if not (0.0 <= self.block_threshold <= 1.0):
            raise ValueError(f"block_threshold must be between 0 and 1, got {self.block_threshold}")
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

        # Check thresholds
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
            "block_threshold": self.config.block_threshold,
            "step_up_mfa_required": mfa_step_up,
            "manual_review_required": review_required,
            "recommendation": recommendation,
        }
