package llmjudge

import (
	"context"
	"fmt"
	"testing"

	goeval "github.com/datar-psa/go-eval"
)

// mockLLMGenerator is a simple mock for unit tests
type mockLLMGenerator struct {
	response string
	err      error
}

func (m *mockLLMGenerator) Generate(ctx context.Context, prompt string) (string, error) {
	if m.err != nil {
		return "", m.err
	}
	return m.response, nil
}

func TestFactuality_Unit(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name         string
		llmResponse  string
		llmErr       error
		input        string
		output       string
		expected     string
		wantErr      error
		wantScore    float64
		wantRawScore int
	}{
		{
			name: "fully correct",
			llmResponse: `Let me evaluate this:
1. The expected answer states Paris is the capital of France
2. The actual output also states Paris is the capital of France
3. There are no contradictions

SCORE: 10`,
			input:        "What is the capital of France?",
			output:       "Paris is the capital of France",
			expected:     "Paris",
			wantScore:    1.0,
			wantRawScore: 10,
		},
		{
			name: "partially correct",
			llmResponse: `Let me evaluate this:
1. The expected answer is 4
2. The actual output says approximately 4
3. This is close but not exact

SCORE: 7`,
			input:        "What is 2+2?",
			output:       "approximately 4",
			expected:     "4",
			wantScore:    0.7,
			wantRawScore: 7,
		},
		{
			name: "completely wrong",
			llmResponse: `Let me evaluate this:
1. The expected answer is London
2. The actual output says Paris
3. This is completely wrong

SCORE: 0`,
			input:        "What is the capital of England?",
			output:       "Paris",
			expected:     "London",
			wantScore:    0.0,
			wantRawScore: 0,
		},
		{
			name:      "no expected value",
			input:     "What is 2+2?",
			output:    "4",
			expected:  "",
			wantErr:   goeval.ErrNoExpectedValue,
			wantScore: 0.0,
		},
		{
			name:      "llm error",
			llmErr:    fmt.Errorf("API error"),
			input:     "What is 2+2?",
			output:    "4",
			expected:  "4",
			wantScore: 0.0,
		},
		{
			name:        "invalid score format",
			llmResponse: "This is factually correct but no score provided",
			input:       "What is 2+2?",
			output:      "4",
			expected:    "4",
			wantScore:   0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockLLM := &mockLLMGenerator{
				response: tt.llmResponse,
				err:      tt.llmErr,
			}

			scorer := Factuality(FactualityOptions{LLM: mockLLM})
			result := scorer.Score(ctx, tt.input, tt.output, tt.expected)

			if tt.wantErr != nil {
				if result.Error != tt.wantErr {
					t.Errorf("Factuality.Score() error = %v, wantErr %v", result.Error, tt.wantErr)
				}
			} else if tt.llmErr != nil {
				if result.Error == nil {
					t.Error("Factuality.Score() expected error but got none")
				}
			} else if tt.llmResponse != "" && tt.wantErr == nil {
				if result.Error != nil && tt.name != "invalid score format" {
					t.Errorf("Factuality.Score() unexpected error = %v", result.Error)
				}
			}

			if result.Score != tt.wantScore {
				t.Errorf("Factuality.Score() score = %v, wantScore %v", result.Score, tt.wantScore)
			}

			if tt.wantRawScore > 0 {
				if rawScore, ok := result.Metadata["raw_score"].(int); !ok || rawScore != tt.wantRawScore {
					t.Errorf("Factuality.Score() raw_score = %v, want %v", rawScore, tt.wantRawScore)
				}
			}

			if result.Name != "Factuality" {
				t.Errorf("Factuality.Score() name = %v, want 'Factuality'", result.Name)
			}
		})
	}
}

func TestFactuality_NoLLM(t *testing.T) {
	ctx := context.Background()

	scorer := Factuality(FactualityOptions{})
	result := scorer.Score(ctx, "input", "output", "expected")

	if result.Error == nil {
		t.Error("Factuality.Score() expected error when LLM is nil")
	}

	if result.Score != 0 {
		t.Errorf("Factuality.Score() score = %v, want 0", result.Score)
	}
}

func TestExtractScore(t *testing.T) {
	tests := []struct {
		name          string
		response      string
		wantScore     int
		wantReasonLen int
		wantErr       bool
	}{
		{
			name: "valid score at end",
			response: `Here is my reasoning:
1. First point
2. Second point

SCORE: 8`,
			wantScore:     8,
			wantReasonLen: 40,
		},
		{
			name: "score with whitespace",
			response: `Reasoning here

SCORE:    5`,
			wantScore:     5,
			wantReasonLen: 10,
		},
		{
			name:     "no score",
			response: "Just some text without a score",
			wantErr:  true,
		},
		{
			name:     "invalid score value",
			response: "SCORE: abc",
			wantErr:  true,
		},
		{
			name:     "score out of range high",
			response: "SCORE: 15",
			wantErr:  true,
		},
		{
			name:     "score out of range low",
			response: "SCORE: -1",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score, reasoning, err := extractScore(tt.response)

			if (err != nil) != tt.wantErr {
				t.Errorf("extractScore() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if score != tt.wantScore {
					t.Errorf("extractScore() score = %v, want %v", score, tt.wantScore)
				}
				if len(reasoning) < tt.wantReasonLen {
					t.Errorf("extractScore() reasoning length = %v, want at least %v", len(reasoning), tt.wantReasonLen)
				}
			}
		})
	}
}
