package com.sentinelpay.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record PolicyEvaluationDto(
        String action,
        @JsonProperty("challenge_threshold") Double challengeThreshold,
        @JsonProperty("manual_review_threshold") Double manualReviewThreshold,
        @JsonProperty("block_threshold") Double blockThreshold,
        @JsonProperty("step_up_mfa_required") Boolean stepUpMfaRequired,
        @JsonProperty("manual_review_required") Boolean manualReviewRequired,
        String recommendation
) {
}
