package llmjudge

import (
	"context"
	"testing"

	"github.com/datar-psa/go-eval/internal/testutils"
)

// TestFactuality_Integration tests the Factuality scorer with real Gemini API calls
// This test requires valid Google Cloud credentials and uses hypert to cache requests
func TestFactuality_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ctx := context.Background()

	// Create Gemini generator using test utilities
	llmGen := testutils.NewGeminiGenerator(t, testutils.DefaultGeminiTestConfig("factuality"), "publishers/google/models/gemini-2.5-flash")

	tests := []struct {
		name     string
		input    string
		output   string
		expected string
		minScore float64
		maxScore float64
	}{
		{
			name:     "correct capital answer",
			input:    "What is the capital of France?",
			output:   "Paris",
			expected: "Paris",
			minScore: 0.9,
			maxScore: 1.0,
		},
		{
			name:     "correct math with different wording",
			input:    "What is 2+2?",
			output:   "The answer is 4",
			expected: "4",
			minScore: 0.8,
			maxScore: 1.0,
		},
		{
			name:     "incorrect answer",
			input:    "What is the capital of France?",
			output:   "London",
			expected: "Paris",
			minScore: 0.0,
			maxScore: 0.3,
		},
		{
			name:     "partially correct answer",
			input:    "What is the capital of France?",
			output:   "Paris is a city in France",
			expected: "Paris is the capital of France",
			minScore: 0.4,
			maxScore: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scorer := Factuality(FactualityOptions{LLM: llmGen})
			result := scorer.Score(ctx, tt.input, tt.output, tt.expected)

			if result.Error != nil {
				t.Fatalf("Factuality.Score() unexpected error = %v", result.Error)
			}

			if result.Score < tt.minScore || result.Score > tt.maxScore {
				t.Errorf("Factuality.Score() score = %v, want between %v and %v", result.Score, tt.minScore, tt.maxScore)
				t.Logf("Reasoning: %v", result.Metadata["reasoning"])
				t.Logf("Raw response: %v", result.Metadata["raw_response"])
			}

			if result.Name != "Factuality" {
				t.Errorf("Factuality.Score() name = %v, want 'Factuality'", result.Name)
			}

			// Verify metadata
			if result.Metadata["raw_score"] == nil {
				t.Error("Factuality.Score() missing raw_score in metadata")
			}
			if result.Metadata["reasoning"] == nil {
				t.Error("Factuality.Score() missing reasoning in metadata")
			}
		})
	}
}
