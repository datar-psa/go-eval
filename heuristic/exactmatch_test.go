package heuristic

import (
	"context"
	"testing"

	goeval "github.com/datar-psa/go-eval"
)

func TestExactMatch(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name      string
		opts      ExactMatchOptions
		input     string
		output    string
		expected  string
		wantErr   error
		wantScore float64
	}{
		{
			name:      "exact match",
			opts:      ExactMatchOptions{},
			input:     "What is 2+2?",
			output:    "4",
			expected:  "4",
			wantScore: 1.0,
		},
		{
			name:      "no match",
			opts:      ExactMatchOptions{},
			input:     "What is 2+2?",
			output:    "5",
			expected:  "4",
			wantScore: 0.0,
		},
		{
			name:      "case sensitive mismatch",
			opts:      ExactMatchOptions{CaseInsensitive: false},
			input:     "What is the capital?",
			output:    "Paris",
			expected:  "paris",
			wantScore: 0.0,
		},
		{
			name:      "case insensitive match",
			opts:      ExactMatchOptions{CaseInsensitive: true},
			input:     "What is the capital?",
			output:    "Paris",
			expected:  "paris",
			wantScore: 1.0,
		},
		{
			name:      "whitespace sensitive mismatch",
			opts:      ExactMatchOptions{TrimWhitespace: false},
			input:     "What is 2+2?",
			output:    "4 ",
			expected:  "4",
			wantScore: 0.0,
		},
		{
			name:      "whitespace insensitive match",
			opts:      ExactMatchOptions{TrimWhitespace: true},
			input:     "What is 2+2?",
			output:    "  4  ",
			expected:  "4",
			wantScore: 1.0,
		},
		{
			name:      "combined options match",
			opts:      ExactMatchOptions{CaseInsensitive: true, TrimWhitespace: true},
			input:     "What is the capital?",
			output:    "  PARIS  ",
			expected:  "paris",
			wantScore: 1.0,
		},
		{
			name:      "no expected value",
			opts:      ExactMatchOptions{},
			input:     "What is 2+2?",
			output:    "4",
			expected:  "",
			wantErr:   goeval.ErrNoExpectedValue,
			wantScore: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scorer := ExactMatch(tt.opts)
			result := scorer.Score(ctx, goeval.ScoreInputs{Output: tt.output, Expected: tt.expected})

			if result.Error != tt.wantErr {
				t.Errorf("ExactMatch.Score() error = %v, wantErr %v", result.Error, tt.wantErr)
			}

			if result.Score != tt.wantScore {
				t.Errorf("ExactMatch.Score() score = %v, wantScore %v", result.Score, tt.wantScore)
			}

			if result.Name != "ExactMatch" {
				t.Errorf("ExactMatch.Score() name = %v, want 'ExactMatch'", result.Name)
			}

			// Verify metadata
			if result.Metadata == nil {
				t.Error("ExactMatch.Score() metadata is nil")
			}
		})
	}
}
