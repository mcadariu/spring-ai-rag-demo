package com.example.braggingrights.demo;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.ollama.OllamaChatClient;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.testcontainers.service.connection.ServiceConnection;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.ollama.OllamaContainer;
import org.springframework.core.io.Resource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.util.Optional.empty;
import static java.util.stream.IntStream.range;

@Testcontainers
@SpringBootTest
@Slf4j
public class DemoApplicationTests {

    private static final String POSTGRES = "postgres";
    private static final String LLAMA3 = "llama3";
    private static final String EMBEDDING_MODEL = "nomic-embed-text";

    @Container
    private static final OllamaContainer ollama = new OllamaContainer("ollama/ollama:latest");

    @Container
    @ServiceConnection
    private static final PostgreSQLContainer<?> pgvector = new PostgreSQLContainer<>("pgvector/pgvector:pg16")
            .withUsername(POSTGRES)
            .withPassword(POSTGRES)
            .withDatabaseName(POSTGRES);

    @Autowired
    private VectorStore vectorStore;

    @Autowired
    private OllamaChatClient ollamaChatClient;

    @Value("classpath:/generate-essay.st")
    protected Resource generateEssay;

    @Value("classpath:/generate-saying.st")
    protected Resource generateSaying;

    @Value("classpath:/guess-saying.st")
    protected Resource guessSaying;

    @Test
    void rag_workflow() {
        var sayings = new ArrayList<String>();
        var sayingToEssay = new HashMap<String, String>();
        var documents = new ArrayList<Document>();

        pullModels();

        range(1, 5).forEach(i -> extractContentBetweenQuotes(
                callama(generateSaying, Map.of("sayings", sayings)))
                .ifPresent(sayings::add)
        );

        sayings.forEach(saying -> {
            var essay = callama(generateEssay, Map.of("saying", saying))
                    .replaceAll(saying, "");

            documents.add(new Document(essay));
            sayingToEssay.put(saying, essay);
        });

        vectorStore.add(documents);

        sayingToEssay.forEach((saying, essay) -> {
            log.info("""
                    saying: %s""".formatted(saying));
            retrieveAndGuess(saying).ifPresent(guess -> log.info("""
                    guess: %s""".formatted(guess)));
        });
    }

    private Optional<String> retrieveAndGuess(String saying) {
        var retrievedEssay = vectorStore
                .similaritySearch(
                        SearchRequest
                                .query(saying)
                )
                .getFirst()
                .toString();
        return extractContentBetweenQuotes(callama(guessSaying, Map.of("essay", retrievedEssay)));
    }

    private static void pullModels() {
        try {
            ollama.execInContainer("ollama", "pull", LLAMA3);
            ollama.execInContainer("ollama", "pull", EMBEDDING_MODEL);
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private Optional<String> extractContentBetweenQuotes(String input) {
        Matcher m = Pattern
                .compile("\"(.*)\\.?\"")
                .matcher(input);

        return m.find() ? Optional.of(m.group(1)) : empty();
    }

    private String callama(Resource promptTemplate, Map<String, Object> promptTemplateValues) {
        Object templateValue = promptTemplateValues.values().iterator().next();

        templateValue = switch (templateValue) {
            case String value -> value;
            case List list -> String.join("\n * ", list);
            default -> templateValue;
        };

        return ollamaChatClient
                .withDefaultOptions(OllamaOptions
                        .create()
                        .withModel(LLAMA3)
                )
                .call(
                        new Prompt(
                                new PromptTemplate(promptTemplate,
                                        Map.of(promptTemplateValues.keySet().iterator().next(),
                                                templateValue)
                                )
                                        .createMessage())
                )
                .getResult()
                .getOutput()
                .getContent();
    }

    @DynamicPropertySource
    static void ollamaProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.ai.ollama.base-url", ollama::getEndpoint);
        registry.add("spring.ai.ollama.embedding.options.model", () -> EMBEDDING_MODEL);
    }

    @DynamicPropertySource
    static void pgVectorProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.ai.vectorstore.pgvector.index-type", () -> "HNSW");
        registry.add("spring.ai.vectorstore.pgvector.distance-type", () -> "COSINE_DISTANCE");
        registry.add("spring.ai.vectorstore.pgvector.dimensions", () -> 768);
    }
}
