package embedding

import (
	"context"
	"testing"

	goeval "github.com/datar-psa/go-eval"
	"github.com/datar-psa/go-eval/internal/testutils"
)

// TestEmbeddingSimilarity_Integration tests the EmbeddingSimilarity scorer with real Gemini embeddings API
// This test requires valid Google Cloud credentials and uses hypert to cache requests
func TestEmbeddingSimilarity_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ctx := context.Background()

	// Create Gemini embedder using test utilities
	embedder := testutils.NewGeminiEmbedder(t, testutils.DefaultGeminiTestConfig("embedding"), "text-embedding-005")

	tests := []struct {
		name     string
		input    string
		output   string
		expected string
		minScore float64
		maxScore float64
	}{
		{
			name:     "identical text",
			input:    "context",
			output:   "What is the type of the leave?",
			expected: "What is the type of the leave?",
			minScore: 0.95,
			maxScore: 1.0,
		},
		{
			name:     "semantically similar questions",
			input:    "context",
			output:   "What is the type of the leave?",
			expected: "Please provide type of the leave",
			minScore: 0.85,
			maxScore: 1.0,
		},
		{
			name:     "similar but different phrasing",
			input:    "context",
			output:   "What is the capital of France?",
			expected: "Tell me France's capital city",
			minScore: 0.80,
			maxScore: 1.0,
		},
		{
			name:     "somewhat related",
			input:    "context",
			output:   "What is the weather today?",
			expected: "Tell me about the temperature",
			minScore: 0.60,
			maxScore: 0.95,
		},
		{
			name:     "completely different",
			input:    "context",
			output:   "What is the capital of France?",
			expected: "How do I bake a cake?",
			minScore: 0.0,
			maxScore: 0.70,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scorer := EmbeddingSimilarity(embedder, EmbeddingSimilarityOptions{})

			result := scorer.Score(ctx, goeval.ScoreInputs{Output: tt.output, Expected: tt.expected})

			if result.Error != nil {
				t.Fatalf("EmbeddingSimilarity.Score() unexpected error = %v", result.Error)
			}

			if result.Score < tt.minScore || result.Score > tt.maxScore {
				t.Errorf("EmbeddingSimilarity.Score() score = %v, want between %v and %v", result.Score, tt.minScore, tt.maxScore)
				t.Logf("Cosine similarity: %v", result.Metadata["cosine_similarity"])
				t.Logf("Embedding dimension: %v", result.Metadata["embedding_dim"])
			}

			if result.Name != "EmbeddingSimilarity" {
				t.Errorf("EmbeddingSimilarity.Score() name = %v, want 'EmbeddingSimilarity'", result.Name)
			}

			// Verify metadata
			if result.Metadata["cosine_similarity"] == nil {
				t.Error("EmbeddingSimilarity.Score() missing cosine_similarity in metadata")
			}
			if result.Metadata["embedding_dim"] == nil {
				t.Error("EmbeddingSimilarity.Score() missing embedding_dim in metadata")
			}
		})
	}
}
