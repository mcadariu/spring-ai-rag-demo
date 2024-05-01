package com.example.braggingrights.demo;

import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
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
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.util.Optional.empty;
import static java.util.stream.IntStream.range;

@Testcontainers
@SpringBootTest
@Slf4j
public class DemoApplicationTests {

    private static final String POSTGRES = "postgres";
    private static final String EMBEDDING_MODEL = "nomic-embed-text";

    @Container
    private static final OllamaContainer ollama = new OllamaContainer("ollama/ollama:latest");

    @Container
    @ServiceConnection
    private static final PostgreSQLContainer<?> pgvector = new PostgreSQLContainer<>("pgvector/pgvector:pg16").withUsername(POSTGRES).withPassword(POSTGRES).withDatabaseName(POSTGRES);

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

    @ParameterizedTest
    @ValueSource(strings = {"llama3"})
    void rag_workflow(String model) {
        pullModels(model);

        var sayings = new ArrayList<String>();
        generateSayings(model, sayings);

        var sayingToEssay = new HashMap<String, String>();
        var essays = new ArrayList<Document>();
        generateEssays(model, sayings, essays, sayingToEssay);

        vectorStore.add(essays);

        sayingToEssay.forEach((saying, essay) -> {
            log.info("Generated saying: " + saying);
            log.info("LLM guess: " + retrieveEssayAndGuessSaying(saying, sayingToEssay.keySet(), model));
        });
    }

    private void generateEssays(String model, ArrayList<String> sayings, ArrayList<Document> documents, HashMap<String, String> sayingToEssay) {
        sayings.forEach(saying -> {
            var essay = callama(generateEssay, Map.of("saying", saying), model)
                    .replaceAll(saying, "")
                    .replace("\"", "");

            documents.add(new Document(essay));
            sayingToEssay.put(saying, essay);
        });
    }

    private void generateSayings(String model, ArrayList<String> sayings) {
        range(1, 10)
                .forEach(i ->
                        extractContentBetweenQuotes(
                                callama(generateSaying, Map.of("sayings", sayings), model))
                                .ifPresent(sayings::add)
                );
    }

    private String retrieveEssayAndGuessSaying(String saying, Set<String> sayings, String model) {
        var retrievedEssay = vectorStore.similaritySearch(SearchRequest.query(saying)).getFirst().getContent();
        return callama(guessSaying, Map.of("essay", retrievedEssay, "sayings", sayings), model);
    }

    private static void pullModels(String model) {
        try {
            ollama.execInContainer("ollama", "pull", model);
            ollama.execInContainer("ollama", "pull", EMBEDDING_MODEL);
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private Optional<String> extractContentBetweenQuotes(String input) {
        Matcher m = Pattern.compile("\"(.*)\\.?\"").matcher(input);
        return m.find() ? Optional.of(m.group(1)) : empty();
    }

    private String callama(Resource promptTemplate, Map<String, Object> promptTemplateValues, String model) {
        return ollamaChatClient
                .withDefaultOptions(OllamaOptions.create().withModel(model))
                .call(createPromptFrom(promptTemplate, promptTemplateValues))
                .getResult()
                .getOutput()
                .getContent();
    }

    @NotNull
    private static Prompt createPromptFrom(Resource promptTemplate, Map<String, Object> promptTemplateValues) {
        Map<String, Object> processedTemplateValues = new HashMap<>();

        for (Map.Entry<String, Object> entry : promptTemplateValues.entrySet()) {
            promptTemplateValues.put(entry.getKey(), switch (entry.getValue()) {
                case String value -> value;
                case List list -> String.join("\n * ", list);
                case Set set -> String.join("\n * ", set.stream().toList());
                default -> entry.getValue();
            });
        }

        return new Prompt(
                new PromptTemplate(promptTemplate,
                        processedTemplateValues)
                        .createMessage());
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
