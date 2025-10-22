# goeval

A Go library for fast, automated evaluation of Large Language Model (LLM) outputs, inspired by Braintrust's [autoevals](https://github.com/braintrustdata/autoevals).

## Features

- Simple, consistent scoring API returning scores in [0.0, 1.0]
- LLM-as-a-judge evaluators: factuality, tonality, and moderation
- Heuristic and embedding-based evaluators for speed and semantics
- Structured outputs from LLM judges for debuggability (choices, confidences, evidence)
- Support for Google Vertex AI (Gemini) via a pluggable generator/provider (more providers planned)

## How Scoring Works

All scorers follow a simple design pattern:
- **output**: The actual response from your model
- **expected**: The expected/reference response (optional for some scorers)

The scorer compares `output` against `expected` and returns a score between 0.0 and 1.0, where 1.0 is the best possible score.

## Getting Started

```go
import (
    "context"
    "fmt"

    "github.com/datar-psa/goeval"
    "github.com/datar-psa/goeval/llmjudge"
    "google.golang.org/genai"
)

func main() {
    ctx := context.Background()

    // Create Gemini client
    genaiClient, _ := genai.NewClient(ctx, &genai.ClientConfig{ /* Your config */ })

    // Create Judge wrapper with functional options
    judge := goeval.NewGeminiLLMJudge(
        goeval.WithGenaiClient(genaiClient),
        goeval.WithModelName("publishers/google/models/gemini-2.5-flash"),
    )

    // Create the Factuality scorer without passing LLM each time
    scorer := judge.Factuality(goeval.FactualityOptions{})

    // Score the model output against an expected answer, with the original question as input
    result := scorer.Score(ctx, goeval.ScoreInputs{
        Input:    "What is the capital of France?",
        Output:   "Paris",
        Expected: "Paris",
    })

    if result.Error != nil {
        panic(result.Error)
    }
    fmt.Printf("Score: %.2f, choice=%v\n", result.Score, result.Metadata["choice"])
}
```

## Scorers

### LLM-as-a-Judge Evaluations

Sophisticated evaluations using language models as judges.

**Package:** `github.com/datar-psa/goeval/llmjudge`

| Scorer     | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| Factuality | LLM judge comparing Output vs Expected for factual consistency               |
| Tonality   | LLM judge for professionalism, kindness, clarity, helpfulness (A–E anchors)  |
| Moderation | Content safety via moderation provider; 1.0 safe, 0.0 unsafe                |

### Heuristic Evaluations

Fast, rule-based scorers that don't require LLMs.

**Package:** `github.com/datar-psa/goeval/heuristic`

| Scorer     | Description                                 |
|------------|---------------------------------------------|
| ExactMatch | Simple equality (configurable case/whitespace) |

### Embedding Evaluations

Semantic similarity using vector embeddings.

**Package:** `github.com/datar-psa/goeval/embedding`

| Scorer             | Description                                    |
|--------------------|------------------------------------------------|
| EmbeddingSimilarity | Cosine similarity over embeddings (semantic closeness) |

## Use Cases

### 1) FAQ Answer Accuracy (Factuality)

Evaluate if an assistant's answer matches a knowledge base answer.

```go
judge := goeval.NewGeminiLLMJudge(
    goeval.WithGenaiClient(genaiClient),
    goeval.WithModelName("publishers/google/models/gemini-2.5-flash"),
)
res := judge.Factuality(goeval.FactualityOptions{}).Score(ctx, goeval.ScoreInputs{
    Input:    "What are store hours on Sundays?",
    Output:   "We're open 10am–6pm on Sundays.",
    Expected: "We are open from 10:00 to 18:00 on Sundays.",
})
// res.Score in [0..1]; metadata includes {choice, explanation, raw_response}
```

### 2) Support Reply Tone (Tonality)

Enforce minimum tone quality across all dimensions with a threshold gate.

```go
judge := goeval.NewGeminiLLMJudge(
    goeval.WithGenaiClient(genaiClient),
    goeval.WithModelName("publishers/google/models/gemini-2.5-flash"),
)
tonality := judge.Tonality(goeval.TonalityOptions{
    ProfessionalismWeight: 0.25,
    KindnessWeight:        0.25,
    ClarityWeight:         0.25,
    HelpfulnessWeight:     0.25,
    Threshold:             0.4, // if any used dimension < 0.4, overall score becomes 0
})

res := tonality.Score(ctx, goeval.ScoreInputs{
    Input:  "Customer complaint about delayed shipment",
    Output: "I'm sorry for the delay — here's what we're doing next...",
})
// res.Metadata contains per-dimension choices/scores and applied weights
```

### 3) Chat Moderation (Moderation)

Block unsafe replies and steer away from sensitive topics (e.g., religion/politics).

```go
// Create Google Cloud Language client
langClient, _ := language.NewRESTClient(ctx)

judge := goeval.NewGeminiLLMJudge(
    goeval.WithLanguageClient(langClient),
)
moderation := judge.Moderation(goeval.ModerationOptions{
    Threshold:  0.5,
    Categories: []string{"Toxic", "Derogatory", "Violent", "Insult", "ReligionBelief", "Politics"},
})

res := moderation.Score(ctx, goeval.ScoreInputs{Output: "Let's discuss your religion and political views..."})
// res.Score = 0.0 if unsafe; metadata includes flagged categories and is_safe=false
```

### 4) Intent Similarity (Embeddings)

Group similar user requests or route to the right workflow.

```go
// Create embedding scorer
embedding := goeval.NewGeminiEmbedding(
    goeval.WithGenaiClient(genaiClient),
    goeval.WithModelName("text-embedding-005"),
)
sim := embedding.Similarity(goeval.EmbeddingSimilarityOptions{})

res := sim.Score(ctx, goeval.ScoreInputs{
    Output:   "Reset my password",
    Expected: "I can't log in to my account",
})
// Higher scores indicate closer semantic intent
```

### 5) Exact Match Validation (Heuristic)

Fast validation for exact matches with configurable options.

```go
// Create heuristic scorer
heuristic := goeval.NewHeuristic()
exactMatch := heuristic.ExactMatch(goeval.ExactMatchOptions{
    CaseSensitive: false,
    TrimWhitespace: true,
})

res := exactMatch.Score(ctx, goeval.ScoreInputs{
    Output:   "Paris",
    Expected: "paris",
})
// res.Score = 1.0 for exact match (case-insensitive)
```

## Design Philosophy

The library is designed with flexibility and composability in mind:

- **Client-First Approach**: We accept pre-configured clients (like `*genai.Client`, `*language.Client`) rather than raw credentials or project IDs. This gives you complete control over authentication, retry policies, and other client configurations.

- **Functional Options Pattern**: All constructors use functional options for clean, extensible APIs that grow gracefully over time.

- **Pluggable Providers**: The scoring interfaces are designed to be implemented by any provider, making it easy to add support for new LLM providers or evaluation services.

## Development

### Running Tests

```bash
go test -short              # Unit tests only
go test                     # All tests
UPDATE_TESTS=true go test   # Update integration test cache (LLM requests)
```

### Request Caching

Currently we're using [hypert](https://github.com/areknoster/hypert) to cache LLM requests. The library's integration tests already demonstrate this pattern.

### Roadmap

- **More Scorers**: Additional evaluation methods
- **Request Caching**: Built-in caching layer for LLM requests (currently one option is hypert)
- **OpenAI Provider**: Native support for OpenAI's GPT models alongside Google Gemini
