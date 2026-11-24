package com.sentinelpay.backend.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;
import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "transactions")
public class Transaction {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "transaction_id", length = 100)
    private String transactionId;

    @Column(nullable = false, length = 20)
    private String type;

    @Column(nullable = false)
    private Double amount;

    @Column(name = "old_balance_org", nullable = false)
    private Double oldbalanceOrg;

    @Column(name = "old_balance_dest", nullable = false)
    private Double oldbalanceDest;

    private Integer step;

    @Column(name = "transaction_day")
    private Integer transactionDay;

    @Column(name = "transaction_hour")
    private Integer transactionHour;

    @Column(name = "fraud_probability", nullable = false)
    private Double fraudProbability;

    @Column(name = "model_threshold", nullable = false)
    private Double modelThreshold;

    @Column(name = "is_above_model_threshold", nullable = false)
    private Boolean isAboveModelThreshold;

    @Column(name = "model_version", nullable = false, length = 64)
    private String modelVersion;

    @Column(nullable = false, length = 20)
    private String action;

    @Column(name = "step_up_mfa_required", nullable = false)
    private Boolean stepUpMfaRequired;

    @Column(name = "manual_review_required", nullable = false)
    private Boolean manualReviewRequired;

    @Column(length = 512)
    private String recommendation;

    @Column(name = "ml_latency_ms")
    private Double mlLatencyMs;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    @PrePersist
    protected void onCreate() {
        this.createdAt = Instant.now();
    }

    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }

    public String getTransactionId() { return transactionId; }
    public void setTransactionId(String transactionId) { this.transactionId = transactionId; }

    public String getType() { return type; }
    public void setType(String type) { this.type = type; }

    public Double getAmount() { return amount; }
    public void setAmount(Double amount) { this.amount = amount; }

    public Double getOldbalanceOrg() { return oldbalanceOrg; }
    public void setOldbalanceOrg(Double oldbalanceOrg) { this.oldbalanceOrg = oldbalanceOrg; }

    public Double getOldbalanceDest() { return oldbalanceDest; }
    public void setOldbalanceDest(Double oldbalanceDest) { this.oldbalanceDest = oldbalanceDest; }

    public Integer getStep() { return step; }
    public void setStep(Integer step) { this.step = step; }

    public Integer getTransactionDay() { return transactionDay; }
    public void setTransactionDay(Integer transactionDay) { this.transactionDay = transactionDay; }

    public Integer getTransactionHour() { return transactionHour; }
    public void setTransactionHour(Integer transactionHour) { this.transactionHour = transactionHour; }

    public Double getFraudProbability() { return fraudProbability; }
    public void setFraudProbability(Double fraudProbability) { this.fraudProbability = fraudProbability; }

    public Double getModelThreshold() { return modelThreshold; }
    public void setModelThreshold(Double modelThreshold) { this.modelThreshold = modelThreshold; }

    public Boolean getIsAboveModelThreshold() { return isAboveModelThreshold; }
    public void setIsAboveModelThreshold(Boolean isAboveModelThreshold) { this.isAboveModelThreshold = isAboveModelThreshold; }

    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }

    public String getAction() { return action; }
    public void setAction(String action) { this.action = action; }

    public Boolean getStepUpMfaRequired() { return stepUpMfaRequired; }
    public void setStepUpMfaRequired(Boolean stepUpMfaRequired) { this.stepUpMfaRequired = stepUpMfaRequired; }

    public Boolean getManualReviewRequired() { return manualReviewRequired; }
    public void setManualReviewRequired(Boolean manualReviewRequired) { this.manualReviewRequired = manualReviewRequired; }

    public String getRecommendation() { return recommendation; }
    public void setRecommendation(String recommendation) { this.recommendation = recommendation; }

    public Double getMlLatencyMs() { return mlLatencyMs; }
    public void setMlLatencyMs(Double mlLatencyMs) { this.mlLatencyMs = mlLatencyMs; }

    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}
