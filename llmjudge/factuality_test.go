package llmjudge

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/datar-psa/goeval/api"
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

func (m *mockLLMGenerator) StructuredGenerate(ctx context.Context, prompt string, schema map[string]interface{}) (map[string]interface{}, error) {
	if m.err != nil {
		return nil, m.err
	}

	// Parse the response as JSON for structured responses
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(m.response), &result); err != nil {
		return nil, fmt.Errorf("failed to parse mock response as JSON: %w", err)
	}
	return result, nil
}

func TestFactuality_Unit(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name            string
		llmResponse     string
		llmErr          error
		input           string
		output          string
		expected        string
		wantErr         error
		wantScore       float64
		wantChoice      string
		wantExplanation string
	}{
		{
			name:            "fully correct",
			llmResponse:     `{"choice": "A", "explanation": "The submitted answer contains all the same details as the expert answer. Both state that Paris is the capital of France with no contradictions."}`,
			input:           "What is the capital of France?",
			output:          "Paris is the capital of France",
			expected:        "Paris",
			wantScore:       1.0,
			wantChoice:      "A",
			wantExplanation: "The submitted answer contains all the same details as the expert answer. Both state that Paris is the capital of France with no contradictions.",
		},
		{
			name:            "partially correct",
			llmResponse:     `{"choice": "D", "explanation": "The submitted answer is a subset of the expert answer and is fully consistent with it. 'approximately 4' is consistent with '4' but provides less precision."}`,
			input:           "What is 2+2?",
			output:          "approximately 4",
			expected:        "4",
			wantScore:       0.4,
			wantChoice:      "D",
			wantExplanation: "The submitted answer is a subset of the expert answer and is fully consistent with it. 'approximately 4' is consistent with '4' but provides less precision.",
		},
		{
			name:            "completely wrong",
			llmResponse:     `{"choice": "E", "explanation": "There is a disagreement between the submitted answer and the expert answer. The expert answer states London is the capital of England, but the submission says Paris."}`,
			input:           "What is the capital of England?",
			output:          "Paris",
			expected:        "London",
			wantScore:       0.0,
			wantChoice:      "E",
			wantExplanation: "There is a disagreement between the submitted answer and the expert answer. The expert answer states London is the capital of England, but the submission says Paris.",
		},
		{
			name:      "no expected value",
			input:     "What is 2+2?",
			output:    "4",
			expected:  "",
			wantErr:   api.ErrNoExpectedValue,
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
			name:        "invalid JSON response",
			llmResponse: "This is not valid JSON",
			input:       "What is 2+2?",
			output:      "4",
			expected:    "4",
			wantScore:   0.0,
		},
		{
			name:        "missing choice field",
			llmResponse: `{"explanation": "This response is missing the choice field"}`,
			input:       "What is 2+2?",
			output:      "4",
			expected:    "4",
			wantScore:   0.0,
		},
		{
			name:        "missing explanation field",
			llmResponse: `{"choice": "C"}`,
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

			scorer := Factuality(mockLLM, FactualityOptions{})
			result := scorer.Score(ctx, api.ScoreInputs{Input: tt.input, Output: tt.output, Expected: tt.expected})

			if tt.wantErr != nil {
				if result.Error != tt.wantErr {
					t.Errorf("Factuality.Score() error = %v, wantErr %v", result.Error, tt.wantErr)
				}
			} else if tt.llmErr != nil {
				if result.Error == nil {
					t.Error("Factuality.Score() expected error but got none")
				}
			} else if tt.llmResponse != "" && tt.wantErr == nil {
				if result.Error != nil && tt.name != "invalid JSON response" && tt.name != "missing choice field" && tt.name != "missing explanation field" {
					t.Errorf("Factuality.Score() unexpected error = %v", result.Error)
				}
			}

			if result.Score != tt.wantScore {
				t.Errorf("Factuality.Score() score = %v, wantScore %v", result.Score, tt.wantScore)
			}

			if tt.wantChoice != "" {
				if choice, ok := result.Metadata["choice"].(string); !ok || choice != tt.wantChoice {
					t.Errorf("Factuality.Score() choice = %v, want %v", choice, tt.wantChoice)
				}
			}

			if tt.wantExplanation != "" {
				if explanation, ok := result.Metadata["explanation"].(string); !ok || explanation != tt.wantExplanation {
					t.Errorf("Factuality.Score() explanation = %v, want %v", explanation, tt.wantExplanation)
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

	scorer := Factuality(nil, FactualityOptions{})
	result := scorer.Score(ctx, api.ScoreInputs{Input: "input", Output: "output", Expected: "expected"})

	if result.Error == nil {
		t.Error("Factuality.Score() expected error when LLM is nil")
	}

	if result.Score != 0 {
		t.Errorf("Factuality.Score() score = %v, want 0", result.Score)
	}
}
