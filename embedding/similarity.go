package embedding

import (
	"context"
	"fmt"
	"math"

	goeval "github.com/datar-psa/go-eval"
	"github.com/datar-psa/go-eval/interfaces"
)

// EmbeddingSimilarityOptions configures the EmbeddingSimilarity scorer
type EmbeddingSimilarityOptions struct {
	// Embedder is used to generate embeddings for text
	Embedder interfaces.Embedder
}

// EmbeddingSimilarity returns a scorer that measures semantic similarity using embeddings
// It computes cosine similarity between the output and expected text embeddings
func EmbeddingSimilarity(opts EmbeddingSimilarityOptions) goeval.Scorer {
	return &embeddingSimilarityScorer{opts: opts}
}

type embeddingSimilarityScorer struct {
	opts EmbeddingSimilarityOptions
}

func (s *embeddingSimilarityScorer) Score(ctx context.Context, input, output, expected string) goeval.Score {
	result := goeval.Score{
		Name:     "EmbeddingSimilarity",
		Metadata: make(map[string]any),
	}

	if expected == "" {
		result.Error = goeval.ErrNoExpectedValue
		result.Score = 0
		return result
	}

	if s.opts.Embedder == nil {
		result.Error = fmt.Errorf("embedder is required")
		result.Score = 0
		return result
	}

	// Generate embeddings
	outputEmbed, err := s.opts.Embedder.Embed(ctx, output)
	if err != nil {
		result.Error = fmt.Errorf("failed to embed output: %w", err)
		result.Score = 0
		return result
	}

	expectedEmbed, err := s.opts.Embedder.Embed(ctx, expected)
	if err != nil {
		result.Error = fmt.Errorf("failed to embed expected: %w", err)
		result.Score = 0
		return result
	}

	// Calculate cosine similarity
	similarity := cosineSimilarity(outputEmbed, expectedEmbed)

	// Normalize from [-1, 1] to [0, 1]
	// In practice, embeddings are usually positive, so similarity is typically in [0, 1]
	// But we handle the full range for robustness
	normalizedScore := (similarity + 1.0) / 2.0
	if normalizedScore < 0 {
		normalizedScore = 0
	}
	if normalizedScore > 1 {
		normalizedScore = 1
	}

	result.Score = normalizedScore
	result.Metadata["cosine_similarity"] = similarity
	result.Metadata["embedding_dim"] = len(outputEmbed)

	return result
}

// cosineSimilarity computes the cosine similarity between two vectors
// Returns a value between -1 and 1, where 1 means identical direction
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (normA * normB)
}
