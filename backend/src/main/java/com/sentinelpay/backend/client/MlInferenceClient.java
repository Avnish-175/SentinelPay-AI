package com.sentinelpay.backend.client;

import com.sentinelpay.backend.dto.MlPredictionResponse;
import com.sentinelpay.backend.dto.TransactionRequest;
import com.sentinelpay.backend.exception.MlServiceUnavailableException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientException;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.time.Duration;

@Component
public class MlInferenceClient {

    private static final Logger log = LoggerFactory.getLogger(MlInferenceClient.class);

    private final WebClient webClient;
    private final Duration timeout;

    public MlInferenceClient(
            WebClient mlEngineWebClient,
            @Value("${ml.engine.timeout-ms:5000}") long timeoutMs
    ) {
        this.webClient = mlEngineWebClient;
        this.timeout = Duration.ofMillis(timeoutMs);
    }

    public MlPredictionResponse predict(TransactionRequest request) {
        log.debug("Sending transaction_id={} to FastAPI ML engine /api/v1/predict", request.transactionId());

        int maxAttempts = 3;
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                MlPredictionResponse response = webClient.post()
                        .uri("/api/v1/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(request)
                        .retrieve()
                        .bodyToMono(MlPredictionResponse.class)
                        .timeout(timeout)
                        .block();

                validateResponse(response);
                return response;

            } catch (WebClientResponseException ex) {
                log.error("ML engine returned HTTP {} for transaction_id={}: {}",
                        ex.getStatusCode(), request.transactionId(), ex.getResponseBodyAsString());
                throw new MlServiceUnavailableException(
                        "ML inference service returned error status: " + ex.getStatusCode(), ex
                );
            } catch (WebClientException ex) {
                log.warn("Attempt {}/{} failed to connect to ML engine for transaction_id={}: {}",
                        attempt, maxAttempts, request.transactionId(), ex.getMessage());
                if (attempt == maxAttempts) {
                    throw new MlServiceUnavailableException(
                            "ML inference service is unreachable after " + maxAttempts + " attempts: " + ex.getMessage(), ex
                    );
                }
                try {
                    Thread.sleep(500L * attempt);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new MlServiceUnavailableException("Interrupted while waiting to retry ML engine", ie);
                }
            } catch (MlServiceUnavailableException ex) {
                throw ex;
            } catch (Exception ex) {
                log.error("Unexpected error calling ML engine for transaction_id={}: {}",
                        request.transactionId(), ex.getMessage());
                throw new MlServiceUnavailableException(
                        "ML inference failed: " + ex.getMessage(), ex
                );
            }
        }
        throw new MlServiceUnavailableException("ML inference service is unreachable");
    }

    private void validateResponse(MlPredictionResponse response) {
        if (response == null
                || response.fraudProbability() == null
                || response.policy() == null
                || response.policy().action() == null) {
            log.error("Received invalid or incomplete response from ML engine: {}", response);
            throw new MlServiceUnavailableException("ML inference service returned an invalid or incomplete response");
        }
    }
}
