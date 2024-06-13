package com.example.braggingrights.demo;

import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.ollama.OllamaChatModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Import;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.ollama.OllamaContainer;
import org.springframework.core.io.Resource;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
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
@Import(RagDemoApplicationConfiguration.class)
@Slf4j
public class RAGDemoApplicationTests {

    private static final String EMBEDDING_MODEL = "nomic-embed-text";

    @Value("classpath:/generate-essay.st")
    protected Resource generateEssay;
    public static final String SAYING_PARAMETER_NAME = "saying";

    @Value("classpath:/generate-saying.st")
    protected Resource generateSaying;
    public static final String SAYINGS_PARAMETER_NAME = "sayings";

    @Value("classpath:/guess-saying.st")
    protected Resource guessSaying;
    public static final String ESSAY_PARAMETER_NAME = "essay";

    @Autowired
    private VectorStore vectorStore;

    @Autowired
    private OllamaChatModel chatModel;

    @Autowired
    private OllamaContainer ollama;

    @ParameterizedTest
    @ValueSource(strings = {"llama3"})
    void RAG_workflow(String model) {
        pullModels(model);

        var sayings = new ArrayList<String>();
        generateSayings(model, sayings);

        var sayingToEssay = new HashMap<String, String>();
        var essays = new ArrayList<Document>();
        generateEssays(model, sayings, essays, sayingToEssay);

        Instant before = Instant.now();
        vectorStore.add(essays);
        Instant after = Instant.now();
        log.info("storing the vector documents:" + Duration.between(before, after).toMillis());

        sayingToEssay.forEach((saying, essay) -> {
            log.info("Generated saying: " + saying);
            log.info("LLM guess: " + retrieveEssayAndGuessSaying(saying, sayingToEssay.keySet(), model));
        });

        while(true) { }
    }

    private void generateEssays(String model, ArrayList<String> sayings, ArrayList<Document> documents, HashMap<String, String> sayingToEssay) {
        sayings.forEach(saying -> {
            var essay = callama(generateEssay, Map.of(SAYING_PARAMETER_NAME, saying), model)
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
                                callama(generateSaying, Map.of(SAYINGS_PARAMETER_NAME, sayings), model))
                                .ifPresent(sayings::add)
                );
    }

    private String retrieveEssayAndGuessSaying(String saying, Set<String> sayings, String model) {
        Instant before = Instant.now();
        var retrievedEssay = vectorStore.similaritySearch(SearchRequest.query(saying)).getFirst().getContent();
        Instant after = Instant.now();
        log.info("performing similarity search" + Duration.between(before, after).toMillis());

        return callama(guessSaying,
                Map.of(ESSAY_PARAMETER_NAME, retrievedEssay, SAYINGS_PARAMETER_NAME, sayings),
                model
        );
    }

    private void pullModels(String model) {
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

    private String callama(Resource prompt, Map<String, Object> values, String model) {
        return this.chatModel
                .withModel(model)
                .call(createPromptFrom(prompt, values))
                .getResult()
                .getOutput()
                .getContent();
    }

    @NotNull
    private static Prompt createPromptFrom(Resource promptTemplate, Map<String, Object> promptTemplateValues) {
        Map<String, Object> processedTemplateValues = new HashMap<>();

        promptTemplateValues.forEach((key, v) -> {
            processedTemplateValues.put(key, switch (v) {
                case List list -> String.join("\n * ", list);
                case Set set -> String.join("\n * ", set.stream().toList());
                default -> v;
            });
        });

        return new Prompt(
                new PromptTemplate(promptTemplate,
                        processedTemplateValues)
                        .createMessage());
    }

    @DynamicPropertySource
    static void ollamaProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.ai.ollama.embedding.options.model", () -> EMBEDDING_MODEL);
    }

    @DynamicPropertySource
    static void pgVectorProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.ai.vectorstore.pgvector.index-type", () -> "HNSW");
        registry.add("spring.ai.vectorstore.pgvector.distance-type", () -> "COSINE_DISTANCE");
        registry.add("spring.ai.vectorstore.pgvector.dimensions", () -> 768);
    }
}
