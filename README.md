# go-eval

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

    goeval "github.com/datar-psa/go-eval"
    "github.com/datar-psa/go-eval/gemini"
    "github.com/datar-psa/go-eval/llmjudge"
)

func main() {
    ctx := context.Background()

    // Create LLM generator (Vertex AI Gemini shown as an example)
    gen := gemini.NewGenerator(genaiClient, "publishers/google/models/gemini-2.5-flash")

    // Create the Factuality scorer
    scorer := llmjudge.Factuality(gen, llmjudge.FactualityOptions{})

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

**Package:** `github.com/datar-psa/go-eval/llmjudge`

| Scorer     | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| Factuality | LLM judge comparing Output vs Expected for factual consistency               |
| Tonality   | LLM judge for professionalism, kindness, clarity, helpfulness (A–E anchors)  |
| Moderation | Content safety via moderation provider; 1.0 safe, 0.0 unsafe                |

### Heuristic Evaluations

Fast, rule-based scorers that don't require LLMs.

**Package:** `github.com/datar-psa/go-eval/heuristic`

| Scorer     | Description                                 |
|------------|---------------------------------------------|
| ExactMatch | Simple equality (configurable case/whitespace) |

### Embedding Evaluations

Semantic similarity using vector embeddings.

**Package:** `github.com/datar-psa/go-eval/embedding`

| Scorer             | Description                                    |
|--------------------|------------------------------------------------|
| EmbeddingSimilarity | Cosine similarity over embeddings (semantic closeness) |

## Use Cases

### 1) FAQ Answer Accuracy (Factuality)

Evaluate if an assistant’s answer matches a knowledge base answer.

```go
scorer := llmjudge.Factuality(gen, llmjudge.FactualityOptions{})
res := scorer.Score(ctx, goeval.ScoreInputs{
    Input:    "What are store hours on Sundays?",
    Output:   "We're open 10am–6pm on Sundays.",
    Expected: "We are open from 10:00 to 18:00 on Sundays.",
})
// res.Score in [0..1]; metadata includes {choice, explanation, raw_response}
```

### 2) Support Reply Tone (Tonality)

Enforce minimum tone quality across all dimensions with a threshold gate.

```go
tonality := llmjudge.Tonality(gen, llmjudge.TonalityOptions{
    ProfessionalismWeight: 0.25,
    KindnessWeight:        0.25,
    ClarityWeight:         0.25,
    HelpfulnessWeight:     0.25,
    Threshold:             0.4, // if any used dimension < 0.4, overall score becomes 0
})

res := tonality.Score(ctx, goeval.ScoreInputs{
    Input:  "Customer complaint about delayed shipment",
    Output: "I’m sorry for the delay — here’s what we’re doing next...",
})
// res.Metadata contains per-dimension choices/scores and applied weights
```

### 3) Chat Moderation (Moderation)

Block unsafe replies and steer away from sensitive topics (e.g., religion/politics).

```go
provider := gemini.NewGoogleCloudProvider(gemini.GoogleCloudOptions{ /* http client + project */ })
moderation := llmjudge.Moderation(provider, llmjudge.ModerationOptions{
    Threshold:  0.5,
    Categories: []string{"Toxic", "Derogatory", "Violent", "Insult", "ReligionBelief", "Politics"},
})

res := moderation.Score(ctx, goeval.ScoreInputs{Output: "Let’s discuss your religion and political views..."})
// res.Score = 0.0 if unsafe; metadata includes flagged categories and is_safe=false
```

### 4) Intent Similarity (Embeddings)

Group similar user requests or route to the right workflow.

```go
sim := embedding.EmbeddingSimilarity(embedding.EmbeddingSimilarityOptions{Embedder: embedder})
res := sim.Score(ctx, goeval.ScoreInputs{
    Output:   "Reset my password",
    Expected: "I can’t log in to my account",
})
// Higher scores indicate closer semantic intent
```

## Running Tests

```bash
go test -short              # Unit tests only
go test                     # All tests
UPDATE_TESTS=true go test   # Update integration test cache (LLM requests)
```
