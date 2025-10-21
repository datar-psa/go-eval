package llmjudge

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	goeval "github.com/datar-psa/go-eval"
)

// mockLLMGeneratorRubric is a simple mock for unit tests
type mockLLMGeneratorRubric struct {
	response string
	err      error
}

func (m *mockLLMGeneratorRubric) Generate(ctx context.Context, prompt string) (string, error) {
	if m.err != nil {
		return "", m.err
	}
	return m.response, nil
}

func (m *mockLLMGeneratorRubric) StructuredGenerate(ctx context.Context, prompt string, schema map[string]interface{}) (map[string]interface{}, error) {
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

func TestToneRubric_Unit(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                  string
		llmResponse           string
		llmErr                error
		input                 string
		output                string
		expected              string
		weights               [4]float64
		wantErr               error
		wantScore             float64
		wantProfChoice        string
		wantKindChoice        string
		wantClarityChoice     string
		wantHelpfulnessChoice string
		wantProfScore         float64
		wantKindScore         float64
		wantClarityScore      float64
		wantHelpfulnessScore  float64
	}{
		{
			name:                  "excellent all dimensions",
			llmResponse:           `{"professionalism": "E", "kindness": "E", "clarity": "E", "helpfulness": "E"}`,
			input:                 "customer complaint",
			output:                "I understand your frustration and I'm here to help resolve this issue professionally.",
			expected:              "",
			weights:               [4]float64{0.3, 0.2, 0.3, 0.2}, // Custom weights
			wantScore:             1.0,                            // All dimensions excellent
			wantProfChoice:        "E",
			wantKindChoice:        "E",
			wantClarityChoice:     "E",
			wantHelpfulnessChoice: "E",
			wantProfScore:         1.0,
			wantKindScore:         1.0,
			wantClarityScore:      1.0,
			wantHelpfulnessScore:  1.0,
		},
		{
			name:                  "mixed scores",
			llmResponse:           `{"professionalism": "D", "kindness": "E", "clarity": "C", "helpfulness": "D"}`,
			input:                 "support request",
			output:                "I'm really sorry you're experiencing this issue. Let me help you right away.",
			expected:              "",
			weights:               [4]float64{0.3, 0.2, 0.3, 0.2},
			wantScore:             0.725, // 0.3*0.75 + 0.2*1.0 + 0.3*0.5 + 0.2*0.75 = 0.725
			wantProfChoice:        "D",
			wantKindChoice:        "E",
			wantClarityChoice:     "C",
			wantHelpfulnessChoice: "D",
			wantProfScore:         0.75,
			wantKindScore:         1.0,
			wantClarityScore:      0.5,
			wantHelpfulnessScore:  0.75,
		},
		{
			name:                  "default weights",
			llmResponse:           `{"professionalism": "C", "kindness": "D", "clarity": "C", "helpfulness": "D"}`,
			input:                 "question",
			output:                "Answer here",
			expected:              "",
			weights:               [4]float64{0.0, 0.0, 0.0, 0.0}, // Should default to equal weights
			wantScore:             0.625,                          // 0.25*0.5 + 0.25*0.75 + 0.25*0.5 + 0.25*0.75 = 0.625
			wantProfChoice:        "C",
			wantKindChoice:        "D",
			wantClarityChoice:     "C",
			wantHelpfulnessChoice: "D",
			wantProfScore:         0.5,
			wantKindScore:         0.75,
			wantClarityScore:      0.5,
			wantHelpfulnessScore:  0.75,
		},
		{
			name:                  "single dimension only",
			llmResponse:           `{"professionalism": "E", "kindness": "A", "clarity": "A", "helpfulness": "A"}`,
			input:                 "formal inquiry",
			output:                "Professional response",
			expected:              "",
			weights:               [4]float64{1.0, 0.0, 0.0, 0.0}, // Only professionalism matters
			wantScore:             1.0,                            // 1.0*1.0 + 0.0*0.0 + 0.0*0.0 + 0.0*0.0 = 1.0
			wantProfChoice:        "E",
			wantKindChoice:        "A",
			wantClarityChoice:     "A",
			wantHelpfulnessChoice: "A",
			wantProfScore:         1.0,
			wantKindScore:         0.0,
			wantClarityScore:      0.0,
			wantHelpfulnessScore:  0.0,
		},
		{
			name:                  "clarity and helpfulness only",
			llmResponse:           `{"professionalism": "A", "kindness": "A", "clarity": "E", "helpfulness": "E"}`,
			input:                 "educational content",
			output:                "Clear and helpful response",
			expected:              "",
			weights:               [4]float64{0.0, 0.0, 0.5, 0.5}, // Only clarity and helpfulness
			wantScore:             1.0,                            // 0.0*0.0 + 0.0*0.0 + 0.5*1.0 + 0.5*1.0 = 1.0
			wantProfChoice:        "A",
			wantKindChoice:        "A",
			wantClarityChoice:     "E",
			wantHelpfulnessChoice: "E",
			wantProfScore:         0.0,
			wantKindScore:         0.0,
			wantClarityScore:      1.0,
			wantHelpfulnessScore:  1.0,
		},
		{
			name:      "llm error",
			llmErr:    fmt.Errorf("API error"),
			input:     "question",
			output:    "response",
			expected:  "",
			wantScore: 0.0,
		},
		{
			name:        "invalid JSON response",
			llmResponse: "This is not valid JSON",
			input:       "question",
			output:      "response",
			expected:    "",
			wantScore:   0.0,
		},
		{
			name:        "missing dimensions",
			llmResponse: `{"professionalism": "E", "kindness": "E"}`,
			input:       "question",
			output:      "response",
			expected:    "",
			wantScore:   0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockLLM := &mockLLMGeneratorRubric{
				response: tt.llmResponse,
				err:      tt.llmErr,
			}

			scorer := ToneRubric(mockLLM, ToneRubricOptions{
				Weights: tt.weights,
			})

			result := scorer.Score(ctx, goeval.ScoreInputs{Input: tt.input, Output: tt.output, Expected: tt.expected})

			if tt.wantErr != nil {
				if result.Error != tt.wantErr {
					t.Errorf("ToneRubric.Score() error = %v, wantErr %v", result.Error, tt.wantErr)
				}
			} else if tt.llmErr != nil {
				if result.Error == nil {
					t.Error("ToneRubric.Score() expected error but got none")
				}
			} else if tt.llmResponse != "" && tt.wantErr == nil {
				if result.Error != nil && tt.name != "invalid JSON response" && tt.name != "missing dimensions" {
					t.Errorf("ToneRubric.Score() unexpected error = %v", result.Error)
				}
			}

			if result.Score != tt.wantScore {
				t.Errorf("ToneRubric.Score() score = %v, wantScore %v", result.Score, tt.wantScore)
			}

			if tt.wantProfChoice != "" {
				if profChoice, ok := result.Metadata["professionalism.choice"].(string); !ok || profChoice != tt.wantProfChoice {
					t.Errorf("ToneRubric.Score() professionalism.choice = %v, want %v", profChoice, tt.wantProfChoice)
				}
			}

			if tt.wantKindChoice != "" {
				if kindChoice, ok := result.Metadata["kindness.choice"].(string); !ok || kindChoice != tt.wantKindChoice {
					t.Errorf("ToneRubric.Score() kindness.choice = %v, want %v", kindChoice, tt.wantKindChoice)
				}
			}

			if tt.wantClarityChoice != "" {
				if clarityChoice, ok := result.Metadata["clarity.choice"].(string); !ok || clarityChoice != tt.wantClarityChoice {
					t.Errorf("ToneRubric.Score() clarity.choice = %v, want %v", clarityChoice, tt.wantClarityChoice)
				}
			}

			if tt.wantHelpfulnessChoice != "" {
				if helpfulnessChoice, ok := result.Metadata["helpfulness.choice"].(string); !ok || helpfulnessChoice != tt.wantHelpfulnessChoice {
					t.Errorf("ToneRubric.Score() helpfulness.choice = %v, want %v", helpfulnessChoice, tt.wantHelpfulnessChoice)
				}
			}

			if tt.wantProfScore >= 0 {
				if profScore, ok := result.Metadata["professionalism.score"].(float64); !ok || profScore != tt.wantProfScore {
					t.Errorf("ToneRubric.Score() professionalism.score = %v, want %v", profScore, tt.wantProfScore)
				}
			}

			if tt.wantKindScore >= 0 {
				if kindScore, ok := result.Metadata["kindness.score"].(float64); !ok || kindScore != tt.wantKindScore {
					t.Errorf("ToneRubric.Score() kindness.score = %v, want %v", kindScore, tt.wantKindScore)
				}
			}

			if tt.wantClarityScore >= 0 {
				if clarityScore, ok := result.Metadata["clarity.score"].(float64); !ok || clarityScore != tt.wantClarityScore {
					t.Errorf("ToneRubric.Score() clarity.score = %v, want %v", clarityScore, tt.wantClarityScore)
				}
			}

			if tt.wantHelpfulnessScore >= 0 {
				if helpfulnessScore, ok := result.Metadata["helpfulness.score"].(float64); !ok || helpfulnessScore != tt.wantHelpfulnessScore {
					t.Errorf("ToneRubric.Score() helpfulness.score = %v, want %v", helpfulnessScore, tt.wantHelpfulnessScore)
				}
			}

			if result.Name != "ToneRubric" {
				t.Errorf("ToneRubric.Score() name = %v, want 'ToneRubric'", result.Name)
			}
		})
	}
}

func TestToneRubric_NoLLM(t *testing.T) {
	ctx := context.Background()

	scorer := ToneRubric(nil, ToneRubricOptions{})
	result := scorer.Score(ctx, goeval.ScoreInputs{Input: "context", Output: "output", Expected: "expected"})

	if result.Error == nil {
		t.Error("ToneRubric.Score() expected error when LLM is nil")
	}

	if result.Score != 0 {
		t.Errorf("ToneRubric.Score() score = %v, want 0", result.Score)
	}
}
