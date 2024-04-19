package com.example.braggingrights.demo;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.ollama.OllamaChatClient;
import org.springframework.ai.ollama.api.OllamaOptions;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.testcontainers.service.connection.ServiceConnection;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.ollama.OllamaContainer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.util.Optional.empty;
import static java.util.stream.IntStream.range;

@Testcontainers
@SpringBootTest
public class DemoApplicationTests {

    public static final String POSTGRES = "postgres";

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

    @Test
    void rag_workflow() {
        pullModels();

        var sayings = new ArrayList<String>();

        range(1, 10).forEach(i ->
                extractContentBetweenQuotes(
                        callama(createPrompt(sayings))
                ).ifPresent(sayings::add));

        var sayingToEssay = new HashMap<String, String>();
        var docs = new ArrayList<Document>();

        sayings.forEach(saying -> {
            var essay = callama("Write a short essay under 200 words explaining the meaning of the following saying: " + saying)
                    .replaceAll(saying, "");

            docs.add(new Document(essay));
            sayingToEssay.put(saying, essay);
        });

        vectorStore.add(docs);

        sayingToEssay.forEach((saying, essay) -> {
            var retrievedEssay = vectorStore
					.similaritySearch(
							SearchRequest
									.query(saying)
					)
					.getFirst()
					.toString();

            var guess = callama("The essay on the next lines was generated from a famous one-phrase saying. Can you please tell me what is the saying that the essay is based on? Give me the saying only and nothing else. \n" + retrievedEssay);

            System.out.println("Initial saying:" + saying);
			System.out.println("Clean guess:" + guess);
			System.out.println(essay.equals(retrievedEssay));
		});
    }

    private static String createPrompt(List<String> sayingsSet) {
        StringBuilder prompt = new StringBuilder("Give me a one-phrase short random saying containing life advice. Do not include the author. Do not give me any of these options: \n");
        for (String saying : sayingsSet) {
            prompt.append("* \"").append(saying).append("\"\n");
        }
        return prompt.toString();
    }

    private static void pullModels() {
        try {
            ollama.execInContainer("ollama", "pull", "llama3");
            ollama.execInContainer("ollama", "pull", "nomic-embed-text");
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private Optional<String> extractContentBetweenQuotes(String input) {
        Matcher m = Pattern
                .compile("\"(.*).\"")
                .matcher(input);

        return m.find() ? Optional.of(m.group(1)) : empty();
    }

    private String callama(String prompt) {
        return ollamaChatClient
                .withDefaultOptions(OllamaOptions.create().withModel("llama3"))
                .call(new Prompt(prompt))
                .getResult()
                .getOutput()
                .getContent();
    }

    @DynamicPropertySource
    static void ollamaProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.ai.ollama.base-url", ollama::getEndpoint);
        registry.add("spring.ai.ollama.embedding.options.model", () -> "nomic-embed-text");
    }

    @DynamicPropertySource
    static void pgVectorProperties(DynamicPropertyRegistry registry) {
        registry.add("spring.ai.vectorstore.pgvector.index-type", () -> "HNSW");
        registry.add("spring.ai.vectorstore.pgvector.distance-type", () -> "COSINE_DISTANCE");
        registry.add("spring.ai.vectorstore.pgvector.dimensions", () -> 768);
    }
}
