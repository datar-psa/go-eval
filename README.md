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
| `ToneRubric` | Evaluates professionalism, kindness, clarity, and helpfulness using rubric-based scoring |
| `Moderation` | Evaluates content safety using moderation providers (e.g., Google Cloud Natural Language API) |

**Example:**
```go
import (
    "github.com/datar-psa/go-eval/llmjudge"
    "github.com/datar-psa/go-eval/gemini"
)

llmGen := gemini.NewGenerator(genaiClient, "publishers/google/models/gemini-2.5-flash")
scorer := llmjudge.Factuality(llmjudge.FactualityOptions{LLM: llmGen})
```

**ToneRubric Example:**
```go
import (
    "github.com/datar-psa/go-eval/llmjudge"
    "github.com/datar-psa/go-eval/gemini"
)

llmGen := gemini.NewGenerator(genaiClient, "publishers/google/models/gemini-2.5-flash")
scorer := llmjudge.ToneRubric(llmjudge.ToneRubricOptions{
    LLM:                   llmGen,
    ProfessionalismWeight: 0.6, // Optional, defaults to 0.5 if both are 0
    KindnessWeight:        0.4, // Optional, defaults to 0.5 if both are 0
    // Note: If one weight is 0, that dimension is excluded from scoring
})

result := scorer.Score(ctx, "customer complaint", "I understand your frustration...", "")
// result.Score = weighted composite (0.0-1.0)
// result.Metadata["professionalism.choice"] = "D" (A-E)
// result.Metadata["kindness.choice"] = "E" (A-E)
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

### Response Tone
For evaluating the professionalism, kindness, clarity, and helpfulness of responses, use **ToneRubric**:

```go
llm := gemini.NewGenerator(client, "publishers/google/models/gemini-2.5-flash")
scorer := llmjudge.ToneRubric(llmjudge.ToneRubricOptions{
    LLM:     llm,
    Weights: [4]float64{0.3, 0.2, 0.3, 0.2}, // [professionalism, kindness, clarity, helpfulness]
})

// Single dimension scoring examples:
// Professionalism only:
scorer := llmjudge.ToneRubric(llmjudge.ToneRubricOptions{
    LLM:     llm,
    Weights: [4]float64{1.0, 0.0, 0.0, 0.0}, // Only professionalism matters
})

// Kindness only:
scorer := llmjudge.ToneRubric(llmjudge.ToneRubricOptions{
    LLM:     llm,
    Weights: [4]float64{0.0, 1.0, 0.0, 0.0}, // Only kindness matters
})
```

### Content Safety
For evaluating the safety and appropriateness of content, use **Moderation**:

```go
import (
    "github.com/datar-psa/go-eval/gemini"
    "net/http"
)

provider := gemini.NewGoogleCloudProvider(gemini.GoogleCloudOptions{
    HTTPClient: http.DefaultClient,
    ProjectID:  "your-project-id", // or use APIKey instead
})

scorer := llmjudge.Moderation(llmjudge.ModerationOptions{
    ModerationProvider: provider,
    Threshold:          0.5, // Adjust based on your safety requirements
})
```


## Running Tests

```bash
go test -short              # Unit tests only
go test                     # All tests
UPDATE_TESTS=true go test   # Update integration test cache
```
