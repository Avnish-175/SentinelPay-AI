package com.sentinelpay.backend.service;

import com.sentinelpay.backend.client.MlInferenceClient;
import com.sentinelpay.backend.dto.MlPredictionResponse;
import com.sentinelpay.backend.dto.TransactionRequest;
import com.sentinelpay.backend.dto.TransactionResponse;
import com.sentinelpay.backend.entity.Transaction;
import com.sentinelpay.backend.repository.TransactionRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class TransactionService {

    private static final Logger log = LoggerFactory.getLogger(TransactionService.class);

    private final MlInferenceClient mlInferenceClient;
    private final TransactionRepository transactionRepository;

    public TransactionService(
            MlInferenceClient mlInferenceClient,
            TransactionRepository transactionRepository
    ) {
        this.mlInferenceClient = mlInferenceClient;
        this.transactionRepository = transactionRepository;
    }

    @Transactional
    public TransactionResponse processTransaction(TransactionRequest request) {
        log.info("Processing transaction: id={}, type={}, amount={}",
                request.transactionId(), request.type(), request.amount());

        MlPredictionResponse mlResponse = mlInferenceClient.predict(request);

        Transaction entity = new Transaction();
        entity.setTransactionId(mlResponse.transactionId() != null ? mlResponse.transactionId() : request.transactionId());
        entity.setType(request.type());
        entity.setAmount(request.amount());
        entity.setOldbalanceOrg(request.oldbalanceOrg());
        entity.setOldbalanceDest(request.oldbalanceDest());
        entity.setStep(request.step());
        entity.setTransactionDay(request.transactionDay());
        entity.setTransactionHour(request.transactionHour());
        entity.setFraudProbability(mlResponse.fraudProbability());
        entity.setModelThreshold(mlResponse.modelThreshold());
        entity.setIsAboveModelThreshold(mlResponse.isAboveModelThreshold());
        entity.setModelVersion(mlResponse.modelVersion());
        entity.setAction(mlResponse.policy().action());
        entity.setStepUpMfaRequired(Boolean.TRUE.equals(mlResponse.policy().stepUpMfaRequired()));
        entity.setManualReviewRequired(Boolean.TRUE.equals(mlResponse.policy().manualReviewRequired()));
        entity.setRecommendation(mlResponse.policy().recommendation());
        entity.setMlLatencyMs(mlResponse.latencyMs());

        Transaction saved = transactionRepository.save(entity);

        log.info("Transaction saved: dbId={}, transactionId={}, probability={}, action={}",
                saved.getId(), saved.getTransactionId(), saved.getFraudProbability(), saved.getAction());

        return new TransactionResponse(
                saved.getId(),
                saved.getTransactionId(),
                saved.getType(),
                saved.getAmount(),
                saved.getFraudProbability(),
                saved.getModelThreshold(),
                saved.getIsAboveModelThreshold(),
                saved.getModelVersion(),
                saved.getAction(),
                saved.getStepUpMfaRequired(),
                saved.getManualReviewRequired(),
                saved.getRecommendation(),
                saved.getMlLatencyMs(),
                saved.getCreatedAt()
        );
    }
}
