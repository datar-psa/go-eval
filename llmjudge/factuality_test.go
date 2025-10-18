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
		wantRawScore float64
	}{
		{
			name: "fully correct",
			llmResponse: `Let me evaluate this:
1. The expected answer states Paris is the capital of France
2. The actual output also states Paris is the capital of France
3. There are no contradictions

C`,
			input:        "What is the capital of France?",
			output:       "Paris is the capital of France",
			expected:     "Paris",
			wantScore:    1.0,
			wantRawScore: 1.0,
		},
		{
			name: "partially correct",
			llmResponse: `Let me evaluate this:
1. The expected answer is 4
2. The actual output says approximately 4
3. This is close but not exact

A`,
			input:        "What is 2+2?",
			output:       "approximately 4",
			expected:     "4",
			wantScore:    0.4,
			wantRawScore: 0.4,
		},
		{
			name: "completely wrong",
			llmResponse: `Let me evaluate this:
1. The expected answer is London
2. The actual output says Paris
3. This is completely wrong

D`,
			input:        "What is the capital of England?",
			output:       "Paris",
			expected:     "London",
			wantScore:    0.0,
			wantRawScore: 0.0,
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
				if rawScore, ok := result.Metadata["choice"].(string); !ok || rawScore == "" {
					t.Errorf("Factuality.Score() choice = %v, want non-empty", rawScore)
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

func TestExtractChoice(t *testing.T) {
	tests := []struct {
		name       string
		response   string
		wantChoice string
		wantErr    bool
	}{
		{
			name:       "valid choice A",
			response:   "The submitted answer is a subset of the expert answer and is fully consistent with it. A",
			wantChoice: "A",
		},
		{
			name:       "valid choice B",
			response:   "This is a superset answer. B",
			wantChoice: "B",
		},
		{
			name:       "valid choice C",
			response:   "Same details. C",
			wantChoice: "C",
		},
		{
			name:       "valid choice D",
			response:   "There is disagreement. D",
			wantChoice: "D",
		},
		{
			name:       "valid choice E",
			response:   "Differences don't matter. E",
			wantChoice: "E",
		},
		{
			name:     "no choice",
			response: "Just some text without a choice",
			wantErr:  true,
		},
		{
			name:     "invalid choice",
			response: "This is choice F",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			choice, err := extractChoice(tt.response)

			if (err != nil) != tt.wantErr {
				t.Errorf("extractChoice() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if choice != tt.wantChoice {
					t.Errorf("extractChoice() choice = %v, want %v", choice, tt.wantChoice)
				}
			}
		})
	}
}
