package llmjudge

import (
	"context"
	"fmt"
	"testing"
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
			llmResponse:           "PROFESSIONALISM: E\nKINDNESS: E\nCLARITY: E\nHELPFULNESS: E",
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
			llmResponse:           "PROFESSIONALISM: D\nKINDNESS: E\nCLARITY: C\nHELPFULNESS: D",
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
			llmResponse:           "PROFESSIONALISM: C\nKINDNESS: D\nCLARITY: C\nHELPFULNESS: D",
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
			llmResponse:           "PROFESSIONALISM: E\nKINDNESS: A\nCLARITY: A\nHELPFULNESS: A",
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
			llmResponse:           "PROFESSIONALISM: A\nKINDNESS: A\nCLARITY: E\nHELPFULNESS: E",
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
			name:        "invalid response format",
			llmResponse: "This is not in the expected format",
			input:       "question",
			output:      "response",
			expected:    "",
			wantScore:   0.0,
		},
		{
			name:        "missing dimensions",
			llmResponse: "PROFESSIONALISM: E\nKINDNESS: E",
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

			scorer := ToneRubric(ToneRubricOptions{
				LLM:     mockLLM,
				Weights: tt.weights,
			})

			result := scorer.Score(ctx, tt.input, tt.output, tt.expected)

			if tt.wantErr != nil {
				if result.Error != tt.wantErr {
					t.Errorf("ToneRubric.Score() error = %v, wantErr %v", result.Error, tt.wantErr)
				}
			} else if tt.llmErr != nil {
				if result.Error == nil {
					t.Error("ToneRubric.Score() expected error but got none")
				}
			} else if tt.llmResponse != "" && tt.wantErr == nil {
				if result.Error != nil && tt.name != "invalid response format" && tt.name != "missing dimensions" {
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

	scorer := ToneRubric(ToneRubricOptions{})
	result := scorer.Score(ctx, "input", "output", "expected")

	if result.Error == nil {
		t.Error("ToneRubric.Score() expected error when LLM is nil")
	}

	if result.Score != 0 {
		t.Errorf("ToneRubric.Score() score = %v, want 0", result.Score)
	}
}

func TestExtractToneChoices(t *testing.T) {
	tests := []struct {
		name                  string
		response              string
		wantProfChoice        string
		wantKindChoice        string
		wantClarityChoice     string
		wantHelpfulnessChoice string
		wantErr               bool
	}{
		{
			name:                  "valid format",
			response:              "PROFESSIONALISM: D\nKINDNESS: E\nCLARITY: C\nHELPFULNESS: B",
			wantProfChoice:        "D",
			wantKindChoice:        "E",
			wantClarityChoice:     "C",
			wantHelpfulnessChoice: "B",
		},
		{
			name:                  "with whitespace",
			response:              "PROFESSIONALISM:    C\nKINDNESS:   B\nCLARITY:    A\nHELPFULNESS:   D",
			wantProfChoice:        "C",
			wantKindChoice:        "B",
			wantClarityChoice:     "A",
			wantHelpfulnessChoice: "D",
		},
		{
			name:                  "case insensitive",
			response:              "professionalism: A\nkindness: B\nclarity: C\nhelpfulness: D",
			wantProfChoice:        "A",
			wantKindChoice:        "B",
			wantClarityChoice:     "C",
			wantHelpfulnessChoice: "D",
		},
		{
			name:                  "mixed case",
			response:              "Professionalism: E\nKindness: D\nClarity: C\nHelpfulness: B",
			wantProfChoice:        "E",
			wantKindChoice:        "D",
			wantClarityChoice:     "C",
			wantHelpfulnessChoice: "B",
		},
		{
			name:     "missing professionalism",
			response: "KINDNESS: E\nCLARITY: D\nHELPFULNESS: C",
			wantErr:  true,
		},
		{
			name:     "missing kindness",
			response: "PROFESSIONALISM: E\nCLARITY: D\nHELPFULNESS: C",
			wantErr:  true,
		},
		{
			name:     "missing clarity",
			response: "PROFESSIONALISM: E\nKINDNESS: D\nHELPFULNESS: C",
			wantErr:  true,
		},
		{
			name:     "missing helpfulness",
			response: "PROFESSIONALISM: E\nKINDNESS: D\nCLARITY: C",
			wantErr:  true,
		},
		{
			name:     "invalid choice",
			response: "PROFESSIONALISM: X\nKINDNESS: Y\nCLARITY: Z\nHELPFULNESS: W",
			wantErr:  true,
		},
		{
			name:     "no choices",
			response: "Just some text",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			choices, err := extractToneChoices(tt.response)

			if (err != nil) != tt.wantErr {
				t.Errorf("extractToneChoices() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if choices[0] != tt.wantProfChoice {
					t.Errorf("extractToneChoices() profChoice = %v, want %v", choices[0], tt.wantProfChoice)
				}
				if choices[1] != tt.wantKindChoice {
					t.Errorf("extractToneChoices() kindChoice = %v, want %v", choices[1], tt.wantKindChoice)
				}
				if choices[2] != tt.wantClarityChoice {
					t.Errorf("extractToneChoices() clarityChoice = %v, want %v", choices[2], tt.wantClarityChoice)
				}
				if choices[3] != tt.wantHelpfulnessChoice {
					t.Errorf("extractToneChoices() helpfulnessChoice = %v, want %v", choices[3], tt.wantHelpfulnessChoice)
				}
			}
		})
	}
}
