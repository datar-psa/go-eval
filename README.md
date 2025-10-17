# go-eval

A Go library for fast, automated evaluation of Large Language Model (LLM) outputs, inspired by Braintrust's [autoevals](https://github.com/braintrustdata/autoevals).

## Installation

```bash
go get github.com/datar-psa/go-eval
```

## Quick Start

```go
import (
    goeval "github.com/datar-psa/go-eval"
    "github.com/datar-psa/go-eval/heuristic"
)

scorer := heuristic.ExactMatch(heuristic.ExactMatchOptions{
    CaseInsensitive: true,
    TrimWhitespace:  true,
})

result := scorer.Score(ctx, "What is 2+2?", "4", "4")
fmt.Printf("Score: %.2f\n", result.Score) // 1.00
```

## Categories

### Heuristic Evaluations

Fast, rule-based scorers that don't require LLMs.

**Package:** `github.com/datar-psa/go-eval/heuristic`

| Scorer | Description |
|--------|-------------|
| `ExactMatch` | Checks if output exactly matches expected value |

**Example:**
```go
import "github.com/datar-psa/go-eval/heuristic"

scorer := heuristic.ExactMatch(heuristic.ExactMatchOptions{
    CaseInsensitive: true,
    TrimWhitespace:  true,
})
```

### LLM-as-a-Judge Evaluations

Sophisticated evaluations using language models as judges.

**Package:** `github.com/datar-psa/go-eval/llmjudge`

| Scorer | Description |
|--------|-------------|
| `Factuality` | Evaluates factual consistency using chain-of-thought reasoning |

**Example:**
```go
import (
    "github.com/datar-psa/go-eval/llmjudge"
    "github.com/datar-psa/go-eval/gemini"
)

llmGen := gemini.NewGenerator(genaiClient, "publishers/google/models/gemini-2.5-flash")
scorer := llmjudge.Factuality(llmjudge.FactualityOptions{LLM: llmGen})
```

### Embedding Evaluations

Semantic similarity using vector embeddings.

**Package:** `github.com/datar-psa/go-eval/embedding`

| Scorer | Description |
|--------|-------------|
| `EmbeddingSimilarity` | Measures semantic similarity using cosine similarity of embeddings |

**Example:**
```go
import (
    "github.com/datar-psa/go-eval/embedding"
    "github.com/datar-psa/go-eval/gemini"
)

embedder := gemini.NewEmbedder(genaiClient, "text-embedding-005")
scorer := embedding.EmbeddingSimilarity(embedding.EmbeddingSimilarityOptions{
    Embedder: embedder,
})

// Perfect for comparing question similarity
result := scorer.Score(ctx, 
    "context",
    "What is the type of the leave?",
    "Please provide type of the leave",
)
// Score: ~0.9-1.0 (highly similar semantically)
```

## Development

### Testing with HTTP Caching

Use [hypert](https://github.com/areknoster/hypert) to cache LLM requests in tests:

```go
hypertClient := hypert.TestClient(t, false, // false = replay mode
    hypert.WithNamingScheme(namingScheme),
)

genaiClient, _ := genai.NewClient(ctx, &genai.ClientConfig{
    Backend:    genai.BackendVertexAI,
    HTTPClient: hypertClient, // Cached requests
})
```

Update cache: `UPDATE_TESTS=true go test`

## Use Cases

### Question Similarity
For checking if two questions are semantically similar (e.g., "What is the type of the leave?" vs "Please provide type of the leave"), use **Embedding Similarity**:

```go
embedder := gemini.NewEmbedder(client, "text-embedding-005")
scorer := embedding.EmbeddingSimilarity(embedding.EmbeddingSimilarityOptions{
    Embedder: embedder,
})
```

### Answer Accuracy
For checking if an answer is factually correct, use **Factuality**:

```go
llm := gemini.NewGenerator(client, "publishers/google/models/gemini-2.5-flash")
scorer := llmjudge.Factuality(llmjudge.FactualityOptions{LLM: llm})
```

## Running Tests

```bash
go test -short              # Unit tests only
go test                     # All tests
UPDATE_TESTS=true go test   # Update integration test cache
```
