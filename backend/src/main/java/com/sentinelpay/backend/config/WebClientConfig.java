package com.sentinelpay.backend.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class WebClientConfig {

    @Bean
    public WebClient mlEngineWebClient(
            WebClient.Builder builder,
            @Value("${ml.engine.base-url:http://localhost:8000}") String baseUrl
    ) {
        return builder
                .baseUrl(baseUrl)
                .build();
    }
}
