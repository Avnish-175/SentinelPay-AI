package com.sentinelpay.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;

public record TransactionRequest(
        @JsonProperty("transaction_id")
        String transactionId,

        @NotBlank(message = "Transaction type is required")
        @Pattern(
                regexp = "CASH_IN|CASH_OUT|DEBIT|PAYMENT|TRANSFER",
                message = "type must be one of CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER"
        )
        String type,

        @NotNull(message = "amount is required")
        @DecimalMin(value = "0.0", inclusive = true, message = "amount must be >= 0.0")
        Double amount,

        @NotNull(message = "oldbalanceOrg is required")
        @DecimalMin(value = "0.0", inclusive = true, message = "oldbalanceOrg must be >= 0.0")
        @JsonProperty("oldbalanceOrg")
        Double oldbalanceOrg,

        @NotNull(message = "oldbalanceDest is required")
        @DecimalMin(value = "0.0", inclusive = true, message = "oldbalanceDest must be >= 0.0")
        @JsonProperty("oldbalanceDest")
        Double oldbalanceDest,

        @Min(value = 1, message = "step must be >= 1")
        Integer step,

        @Min(value = 0, message = "transaction_day must be between 0 and 30")
        @Max(value = 30, message = "transaction_day must be between 0 and 30")
        @JsonProperty("transaction_day")
        Integer transactionDay,

        @Min(value = 0, message = "transaction_hour must be between 0 and 23")
        @Max(value = 23, message = "transaction_hour must be between 0 and 23")
        @JsonProperty("transaction_hour")
        Integer transactionHour
) {
}
