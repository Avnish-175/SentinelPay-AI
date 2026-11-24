package com.sentinelpay.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record MlPredictionResponse(
        @JsonProperty("transaction_id") String transactionId,
        @JsonProperty("fraud_probability") Double fraudProbability,
        @JsonProperty("model_threshold") Double modelThreshold,
        @JsonProperty("is_above_model_threshold") Boolean isAboveModelThreshold,
        @JsonProperty("model_version") String modelVersion,
        PolicyEvaluationDto policy,
        @JsonProperty("latency_ms") Double latencyMs
) {
}
