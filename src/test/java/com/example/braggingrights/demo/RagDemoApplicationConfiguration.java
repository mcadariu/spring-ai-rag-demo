package com.example.braggingrights.demo;

import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.testcontainers.service.connection.ServiceConnection;
import org.springframework.context.annotation.Bean;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.ollama.OllamaContainer;
import org.testcontainers.utility.DockerImageName;

@TestConfiguration(proxyBeanMethods = false)
class RagDemoApplicationConfiguration {
    private static final String POSTGRES = "postgres";

    @Bean
    @ServiceConnection
    PostgreSQLContainer<?> postgreSQLContainer() {
        return new PostgreSQLContainer<>(DockerImageName
                .parse("timescale/timescaledb-ha:pg16-all")
                .asCompatibleSubstituteFor("postgres")
        )
                .withUsername(POSTGRES)
                .withPassword(POSTGRES)
                .withDatabaseName(POSTGRES)
                .withInitScript("init-script.sql");
    }

    @Bean
    @ServiceConnection
    OllamaContainer ollamaContainer() {
        return new OllamaContainer("ollama/ollama:latest");
    }
}
