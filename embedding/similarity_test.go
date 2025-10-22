package embedding

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/datar-psa/goeval/api"
)

// mockEmbedder is a simple mock for unit tests
type mockEmbedder struct {
	embeddings map[string][]float64
	err        error
}

func (m *mockEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	if m.err != nil {
		return nil, m.err
	}
	if emb, ok := m.embeddings[text]; ok {
		return emb, nil
	}
	// Return a default embedding if not found
	return []float64{1.0, 0.0, 0.0}, nil
}

func TestEmbeddingSimilarity_Unit(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name         string
		embeddings   map[string][]float64
		embedErr     error
		input        string
		output       string
		expected     string
		wantErr      error
		wantMinScore float64
		wantMaxScore float64
	}{
		{
			name: "identical embeddings",
			embeddings: map[string][]float64{
				"hello": {1.0, 0.0, 0.0},
			},
			output:       "hello",
			expected:     "hello",
			wantMinScore: 0.99,
			wantMaxScore: 1.0,
		},
		{
			name: "very similar embeddings",
			embeddings: map[string][]float64{
				"What is the type of leave?":       {1.0, 0.1, 0.0},
				"Please provide type of the leave": {1.0, 0.15, 0.05},
			},
			output:       "What is the type of leave?",
			expected:     "Please provide type of the leave",
			wantMinScore: 0.8,
			wantMaxScore: 1.0,
		},
		{
			name: "orthogonal embeddings",
			embeddings: map[string][]float64{
				"a": {1.0, 0.0, 0.0},
				"b": {0.0, 1.0, 0.0},
			},
			output:       "a",
			expected:     "b",
			wantMinScore: 0.4, // Normalized from 0 to [0,1] range
			wantMaxScore: 0.6,
		},
		{
			name: "opposite embeddings",
			embeddings: map[string][]float64{
				"a": {1.0, 0.0, 0.0},
				"b": {-1.0, 0.0, 0.0},
			},
			output:       "a",
			expected:     "b",
			wantMinScore: 0.0,
			wantMaxScore: 0.1, // Normalized from -1 to [0,1] range
		},
		{
			name:         "no expected value",
			output:       "hello",
			expected:     "",
			wantErr:      api.ErrNoExpectedValue,
			wantMinScore: 0.0,
			wantMaxScore: 0.0,
		},
		{
			name:         "embedder error",
			embedErr:     fmt.Errorf("API error"),
			output:       "hello",
			expected:     "world",
			wantMinScore: 0.0,
			wantMaxScore: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockEmbed := &mockEmbedder{
				embeddings: tt.embeddings,
				err:        tt.embedErr,
			}

			scorer := EmbeddingSimilarity(mockEmbed, EmbeddingSimilarityOptions{})

			result := scorer.Score(ctx, api.ScoreInputs{Output: tt.output, Expected: tt.expected})

			if tt.wantErr != nil {
				if result.Error != tt.wantErr {
					t.Errorf("EmbeddingSimilarity.Score() error = %v, wantErr %v", result.Error, tt.wantErr)
				}
			} else if tt.embedErr != nil {
				if result.Error == nil {
					t.Error("EmbeddingSimilarity.Score() expected error but got none")
				}
			} else {
				if result.Error != nil {
					t.Errorf("EmbeddingSimilarity.Score() unexpected error = %v", result.Error)
				}
			}

			if result.Score < tt.wantMinScore || result.Score > tt.wantMaxScore {
				t.Errorf("EmbeddingSimilarity.Score() score = %v, want between %v and %v", result.Score, tt.wantMinScore, tt.wantMaxScore)
			}

			if result.Name != "EmbeddingSimilarity" {
				t.Errorf("EmbeddingSimilarity.Score() name = %v, want 'EmbeddingSimilarity'", result.Name)
			}
		})
	}
}

func TestEmbeddingSimilarity_NoEmbedder(t *testing.T) {
	ctx := context.Background()

	scorer := EmbeddingSimilarity(nil, EmbeddingSimilarityOptions{})
	result := scorer.Score(ctx, api.ScoreInputs{Output: "output", Expected: "expected"})

	if result.Error == nil {
		t.Error("EmbeddingSimilarity.Score() expected error when Embedder is nil")
	}

	if result.Score != 0 {
		t.Errorf("EmbeddingSimilarity.Score() score = %v, want 0", result.Score)
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name    string
		a       []float64
		b       []float64
		wantSim float64
		epsilon float64
	}{
		{
			name:    "identical vectors",
			a:       []float64{1.0, 0.0, 0.0},
			b:       []float64{1.0, 0.0, 0.0},
			wantSim: 1.0,
			epsilon: 0.001,
		},
		{
			name:    "orthogonal vectors",
			a:       []float64{1.0, 0.0, 0.0},
			b:       []float64{0.0, 1.0, 0.0},
			wantSim: 0.0,
			epsilon: 0.001,
		},
		{
			name:    "opposite vectors",
			a:       []float64{1.0, 0.0, 0.0},
			b:       []float64{-1.0, 0.0, 0.0},
			wantSim: -1.0,
			epsilon: 0.001,
		},
		{
			name:    "similar vectors",
			a:       []float64{1.0, 0.1, 0.0},
			b:       []float64{1.0, 0.15, 0.05},
			wantSim: 0.98, // Approximately
			epsilon: 0.02,
		},
		{
			name:    "different lengths",
			a:       []float64{1.0, 0.0},
			b:       []float64{1.0, 0.0, 0.0},
			wantSim: 0.0,
			epsilon: 0.001,
		},
		{
			name:    "zero vector",
			a:       []float64{0.0, 0.0, 0.0},
			b:       []float64{1.0, 0.0, 0.0},
			wantSim: 0.0,
			epsilon: 0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sim := cosineSimilarity(tt.a, tt.b)
			if math.Abs(sim-tt.wantSim) > tt.epsilon {
				t.Errorf("cosineSimilarity() = %v, want %v (±%v)", sim, tt.wantSim, tt.epsilon)
			}
		})
	}
}
