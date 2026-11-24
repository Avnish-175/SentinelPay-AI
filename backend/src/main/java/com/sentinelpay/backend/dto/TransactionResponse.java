package com.sentinelpay.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.Instant;
import java.util.UUID;

public record TransactionResponse(
        UUID id,
        @JsonProperty("transaction_id") String transactionId,
        String type,
        Double amount,
        @JsonProperty("fraud_probability") Double fraudProbability,
        @JsonProperty("model_threshold") Double modelThreshold,
        @JsonProperty("is_above_model_threshold") Boolean isAboveModelThreshold,
        @JsonProperty("model_version") String modelVersion,
        String action,
        @JsonProperty("step_up_mfa_required") Boolean stepUpMfaRequired,
        @JsonProperty("manual_review_required") Boolean manualReviewRequired,
        String recommendation,
        @JsonProperty("ml_latency_ms") Double mlLatencyMs,
        @JsonProperty("created_at") Instant createdAt
) {
}
