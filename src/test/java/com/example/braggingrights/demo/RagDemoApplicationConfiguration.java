package com.example.braggingrights.demo;

import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.testcontainers.service.connection.ServiceConnection;
import org.springframework.context.annotation.Bean;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.ollama.OllamaContainer;

@TestConfiguration(proxyBeanMethods = false)
public class RagDemoApplicationConfiguration {
    private static final String POSTGRES = "postgres";

    @Bean
    @ServiceConnection
    PostgreSQLContainer<?> postgreSQLContainer() {
        return new PostgreSQLContainer<>("pgvector/pgvector:pg16")
                .withUsername(POSTGRES)
                .withPassword(POSTGRES)
                .withDatabaseName(POSTGRES);
    }

    @Bean
    @ServiceConnection
    OllamaContainer ollamaContainer() {
        return new OllamaContainer("ollama/ollama:latest");
    }
}