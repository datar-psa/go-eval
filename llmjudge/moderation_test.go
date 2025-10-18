package llmjudge

import (
	"context"
	"fmt"
	"testing"

	"github.com/datar-psa/go-eval/interfaces"
)

// mockModerationProvider is a simple mock for unit tests
type mockModerationProvider struct {
	result *interfaces.ModerationResult
	err    error
}

func (m *mockModerationProvider) Moderate(ctx context.Context, content string) (*interfaces.ModerationResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.result, nil
}

func TestModeration_Unit(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name        string
		mockResult  *interfaces.ModerationResult
		mockErr     error
		input       string
		output      string
		expected    string
		threshold   float64
		categories  []string
		wantErr     bool
		wantScore   float64
		wantUnsafe  bool
		wantFlagged map[string]float64
	}{
		{
			name: "safe content",
			mockResult: &interfaces.ModerationResult{
				Categories: []interfaces.ModerationCategory{
					{Name: "Toxic", Confidence: 0.1},
					{Name: "Violent", Confidence: 0.05},
					{Name: "Sexual", Confidence: 0.0},
				},
				IsSafe:        true,
				MaxConfidence: 0.1,
			},
			input:       "context",
			output:      "This is a helpful response",
			expected:    "",
			threshold:   0.5,
			wantScore:   1.0,
			wantUnsafe:  false,
			wantFlagged: map[string]float64{},
		},
		{
			name: "unsafe content",
			mockResult: &interfaces.ModerationResult{
				Categories: []interfaces.ModerationCategory{
					{Name: "Toxic", Confidence: 0.8},
					{Name: "Violent", Confidence: 0.3},
					{Name: "Sexual", Confidence: 0.0},
				},
				IsSafe:        false,
				MaxConfidence: 0.8,
			},
			input:       "context",
			output:      "This is toxic content",
			expected:    "",
			threshold:   0.5,
			wantScore:   0.0,
			wantUnsafe:  true,
			wantFlagged: map[string]float64{"Toxic": 0.8},
		},
		{
			name: "multiple flagged categories",
			mockResult: &interfaces.ModerationResult{
				Categories: []interfaces.ModerationCategory{
					{Name: "Toxic", Confidence: 0.7},
					{Name: "Violent", Confidence: 0.6},
					{Name: "Sexual", Confidence: 0.0},
				},
				IsSafe:        false,
				MaxConfidence: 0.7,
			},
			input:       "context",
			output:      "This is toxic and violent content",
			expected:    "",
			threshold:   0.5,
			wantScore:   0.0,
			wantUnsafe:  true,
			wantFlagged: map[string]float64{"Toxic": 0.7, "Violent": 0.6},
		},
		{
			name: "custom threshold",
			mockResult: &interfaces.ModerationResult{
				Categories: []interfaces.ModerationCategory{
					{Name: "Toxic", Confidence: 0.3},
					{Name: "Violent", Confidence: 0.2},
				},
				IsSafe:        true,
				MaxConfidence: 0.3,
			},
			input:       "context",
			output:      "Somewhat concerning content",
			expected:    "",
			threshold:   0.25,
			wantScore:   1.0, // Content is safe according to provider
			wantUnsafe:  false,
			wantFlagged: map[string]float64{"Toxic": 0.3}, // But flagged by threshold
		},
		{
			name: "specific categories only",
			mockResult: &interfaces.ModerationResult{
				Categories: []interfaces.ModerationCategory{
					{Name: "Toxic", Confidence: 0.8},
					{Name: "Violent", Confidence: 0.6},
					{Name: "Sexual", Confidence: 0.0},
				},
				IsSafe:        false,
				MaxConfidence: 0.8,
			},
			input:       "context",
			output:      "Content with various issues",
			expected:    "",
			threshold:   0.5,
			categories:  []string{"Toxic"}, // Only check Toxic
			wantScore:   0.0,
			wantUnsafe:  true,
			wantFlagged: map[string]float64{"Toxic": 0.8},
		},
		{
			name:      "provider error",
			mockErr:   fmt.Errorf("API error"),
			input:     "context",
			output:    "content",
			expected:  "",
			wantErr:   true,
			wantScore: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockProvider := &mockModerationProvider{
				result: tt.mockResult,
				err:    tt.mockErr,
			}

			scorer := Moderation(ModerationOptions{
				ModerationProvider: mockProvider,
				Threshold:          tt.threshold,
				Categories:         tt.categories,
			})

			result := scorer.Score(ctx, tt.input, tt.output, tt.expected)

			if tt.wantErr {
				if result.Error == nil {
					t.Error("Moderation.Score() expected error but got none")
				}
			} else {
				if result.Error != nil {
					t.Errorf("Moderation.Score() unexpected error = %v", result.Error)
				}
			}

			if result.Score != tt.wantScore {
				t.Errorf("Moderation.Score() score = %v, wantScore %v", result.Score, tt.wantScore)
			}

			if !tt.wantErr {
				if isSafe, ok := result.Metadata["is_safe"].(bool); !ok || isSafe != !tt.wantUnsafe {
					t.Errorf("Moderation.Score() is_safe = %v, want %v", isSafe, !tt.wantUnsafe)
				}

				if tt.wantFlagged != nil {
					if flagged, ok := result.Metadata["flagged_categories"].(map[string]float64); !ok {
						t.Error("Moderation.Score() missing flagged_categories in metadata")
					} else {
						if len(flagged) != len(tt.wantFlagged) {
							t.Errorf("Moderation.Score() flagged categories count = %v, want %v", len(flagged), len(tt.wantFlagged))
						}
						for category, confidence := range tt.wantFlagged {
							if flaggedConfidence, exists := flagged[category]; !exists || flaggedConfidence != confidence {
								t.Errorf("Moderation.Score() flagged[%s] = %v, want %v", category, flaggedConfidence, confidence)
							}
						}
					}
				}
			}

			if result.Name != "Moderation" {
				t.Errorf("Moderation.Score() name = %v, want 'Moderation'", result.Name)
			}
		})
	}
}

func TestModeration_NoProvider(t *testing.T) {
	ctx := context.Background()

	scorer := Moderation(ModerationOptions{})
	result := scorer.Score(ctx, "input", "output", "expected")

	if result.Error == nil {
		t.Error("Moderation.Score() expected error when provider is nil")
	}

	if result.Score != 0 {
		t.Errorf("Moderation.Score() score = %v, want 0", result.Score)
	}
}
