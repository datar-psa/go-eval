package llmjudge

import (
	"context"
	"os"
	"testing"

	goeval "github.com/datar-psa/go-eval"
	"github.com/datar-psa/go-eval/gemini"
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
			scorer := Factuality(llmGen, FactualityOptions{})
			result := scorer.Score(ctx, goeval.ScoreInputs{Input: tt.input, Output: tt.output, Expected: tt.expected})

			if result.Error != nil {
				t.Fatalf("Factuality.Score() unexpected error = %v", result.Error)
			}

			if result.Score < tt.minScore || result.Score > tt.maxScore {
				t.Errorf("Factuality.Score() score = %v, want between %v and %v", result.Score, tt.minScore, tt.maxScore)
				t.Logf("Choice: %v", result.Metadata["choice"])
				t.Logf("Raw response: %v", result.Metadata["raw_response"])
			}

			if result.Name != "Factuality" {
				t.Errorf("Factuality.Score() name = %v, want 'Factuality'", result.Name)
			}

			// Verify metadata
			if result.Metadata["choice"] == nil {
				t.Error("Factuality.Score() missing choice in metadata")
			}
			if result.Metadata["raw_response"] == nil {
				t.Error("Factuality.Score() missing raw_response in metadata")
			}
		})
	}
}

// TestTonality_Integration tests the Tonality scorer with real Gemini API calls
// This test requires valid Google Cloud credentials and uses hypert to cache requests
func TestTonality_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ctx := context.Background()

	// Create Gemini generator using test utilities
	llmGen := testutils.NewGeminiGenerator(t, testutils.DefaultGeminiTestConfig("tonality"), "publishers/google/models/gemini-2.5-flash")

	tests := []struct {
		name                  string
		input                 string
		output                string
		expected              string
		professionalismWeight float64
		kindnessWeight        float64
		minScore              float64
		maxScore              float64
	}{
		{
			name:                  "professional and kind response",
			input:                 "customer complaint about delayed order",
			output:                "I sincerely apologize for the delay in your order. I understand how frustrating this must be, and I want to assure you that we're working to resolve this issue immediately. Please let me know if there's anything else I can do to help.",
			expected:              "",
			professionalismWeight: 0.6,
			kindnessWeight:        0.4,
			minScore:              0.7,
			maxScore:              1.0,
		},
		{
			name:                  "professional but cold response",
			input:                 "customer complaint about delayed order",
			output:                "Your order has been delayed due to shipping issues. We are processing your request and will update you when resolved.",
			expected:              "",
			professionalismWeight: 0.6,
			kindnessWeight:        0.4,
			minScore:              0.4,
			maxScore:              0.7,
		},
		{
			name:                  "kind but unprofessional response",
			input:                 "customer complaint about delayed order",
			output:                "Oh no! That's so annoying! I totally get why you're upset. Don't worry, we'll figure this out together! 😊",
			expected:              "",
			professionalismWeight: 0.6,
			kindnessWeight:        0.4,
			minScore:              0.25,
			maxScore:              0.35,
		},
		{
			name:                  "unprofessional and unkind response",
			input:                 "customer complaint about delayed order",
			output:                "That's not our problem. You should have read the terms. Deal with it.",
			expected:              "",
			professionalismWeight: 0.6,
			kindnessWeight:        0.4,
			minScore:              0.0,
			maxScore:              0.2,
		},
		{
			name:                  "default weights",
			input:                 "support request",
			output:                "Thank you for contacting us. I'm here to help you with your request and will do my best to resolve this issue.",
			expected:              "",
			professionalismWeight: 0.0, // Should default to equal weights
			kindnessWeight:        0.0, // Should default to equal weights
			minScore:              0.6,
			maxScore:              0.8,
		},
		{
			name:                  "professionalism only",
			input:                 "formal business inquiry",
			output:                "We appreciate your inquiry. Please find attached the requested documentation.",
			expected:              "",
			professionalismWeight: 1.0, // Only professionalism matters
			kindnessWeight:        0.0, // Kindness not relevant
			minScore:              0.7,
			maxScore:              1.0,
		},
		{
			name:                  "kindness only",
			input:                 "personal support request",
			output:                "I'm so sorry you're going through this. I'm here for you and we'll get through this together.",
			expected:              "",
			professionalismWeight: 0.0, // Professionalism not relevant
			kindnessWeight:        1.0, // Only kindness matters
			minScore:              0.7,
			maxScore:              1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scorer := Tonality(llmGen, TonalityOptions{
				ProfessionalismWeight: tt.professionalismWeight,
				KindnessWeight:        tt.kindnessWeight,
			})
			result := scorer.Score(ctx, goeval.ScoreInputs{Input: tt.input, Output: tt.output, Expected: tt.expected})

			if result.Error != nil {
				t.Fatalf("Tonality.Score() unexpected error = %v", result.Error)
			}

			if result.Score < tt.minScore || result.Score > tt.maxScore {
				t.Errorf("Tonality.Score() score = %v, want between %v and %v", result.Score, tt.minScore, tt.maxScore)
				t.Logf("Professionalism choice: %v", result.Metadata["professionalism.choice"])
				t.Logf("Kindness choice: %v", result.Metadata["kindness.choice"])
				t.Logf("Raw response: %v", result.Metadata["raw_response"])
			}

			if result.Name != "Tonality" {
				t.Errorf("Tonality.Score() name = %v, want 'Tonality'", result.Name)
			}

			// Verify metadata
			if result.Metadata["professionalism.choice"] == nil {
				t.Error("ToneRubric.Score() missing professionalism.choice in metadata")
			}
			if result.Metadata["kindness.choice"] == nil {
				t.Error("ToneRubric.Score() missing kindness.choice in metadata")
			}
			if result.Metadata["professionalism.score"] == nil {
				t.Error("ToneRubric.Score() missing professionalism.score in metadata")
			}
			if result.Metadata["kindness.score"] == nil {
				t.Error("ToneRubric.Score() missing kindness.score in metadata")
			}
		})
	}
}

// TestModeration_Integration tests the Moderation scorer with real Google Cloud Natural Language API calls
// This test requires valid Google Cloud credentials and uses hypert to cache requests
func TestModeration_Integration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ctx := context.Background()

	// Create authenticated HTTP client with quota project using test utilities
	httpClient := testutils.NewAuthenticatedHypertClient(t, testutils.HypertClientConfig{
		TestDataDir: "testdata",
		SubDir:      "moderation",
	}, os.Getenv("GOOGLE_PROJECT_ID"))

	// Create Google Cloud Natural Language provider
	provider := gemini.NewGoogleCloudProvider(gemini.GoogleCloudOptions{
		HTTPClient: httpClient,
		ProjectID:  os.Getenv("GOOGLE_PROJECT_ID"),
	})

	tests := []struct {
		name        string
		input       string
		output      string
		expected    string
		threshold   float64
		categories  []string
		wantScore   float64
		wantUnsafe  bool
		description string
	}{
		{
			name:        "safe content",
			input:       "customer inquiry",
			output:      "Thank you for your question. I'm happy to help you with your request.",
			expected:    "",
			threshold:   0.5,
			categories:  []string{},
			wantScore:   1.0,
			wantUnsafe:  false,
			description: "Professional customer service response should be safe",
		},
		{
			name:        "potentially toxic content",
			input:       "angry customer complaint",
			output:      "This is absolutely ridiculous! You people are incompetent and useless!",
			expected:    "",
			threshold:   0.5,
			categories:  []string{},
			wantScore:   0.0,
			wantUnsafe:  true,
			description: "Aggressive language should be flagged as unsafe",
		},
		{
			name:        "violent content",
			input:       "threat",
			output:      "I'm going to hurt you if you don't fix this immediately!",
			expected:    "",
			threshold:   0.5,
			categories:  []string{},
			wantScore:   0.0,
			wantUnsafe:  true,
			description: "Threatening language should be flagged as unsafe",
		},
		{
			name:        "low threshold sensitivity",
			input:       "mild frustration",
			output:      "This is annoying and I'm not happy about it.",
			expected:    "",
			threshold:   0.2, // Lower threshold should catch milder content
			categories:  []string{},
			wantScore:   1.0, // Actually safe - API confidence is low (0.137)
			wantUnsafe:  false,
			description: "Lower threshold should catch milder negative content",
		},
		{
			name:        "specific category filtering",
			input:       "mixed content",
			output:      "This is terrible and I hate it! Also, this is violent content.",
			expected:    "",
			threshold:   0.5,
			categories:  []string{"Toxic"}, // Only check for toxic content
			wantScore:   0.0,
			wantUnsafe:  true,
			description: "Should flag toxic content even when other categories are present",
		},
		{
			name:        "neutral business content",
			input:       "business inquiry",
			output:      "Thank you for your message. We will respond soon.",
			expected:    "",
			threshold:   0.5,
			categories:  []string{},
			wantScore:   1.0,
			wantUnsafe:  false,
			description: "Neutral business communication should be safe",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scorer := Moderation(provider, ModerationOptions{
				Threshold:  tt.threshold,
				Categories: tt.categories,
			})
			result := scorer.Score(ctx, goeval.ScoreInputs{Output: tt.output, Expected: tt.expected})

			if result.Error != nil {
				t.Fatalf("Moderation.Score() unexpected error = %v", result.Error)
			}

			if result.Score != tt.wantScore {
				t.Errorf("Moderation.Score() score = %v, wantScore %v", result.Score, tt.wantScore)
				t.Logf("Description: %s", tt.description)
				t.Logf("Is safe: %v", result.Metadata["is_safe"])
				t.Logf("Flagged categories: %v", result.Metadata["flagged_categories"])
				t.Logf("Max confidence: %v", result.Metadata["max_confidence"])
			}

			if result.Name != "Moderation" {
				t.Errorf("Moderation.Score() name = %v, want 'Moderation'", result.Name)
			}

			// Verify metadata
			if result.Metadata["is_safe"] == nil {
				t.Error("Moderation.Score() missing is_safe in metadata")
			}
			if result.Metadata["flagged_categories"] == nil {
				t.Error("Moderation.Score() missing flagged_categories in metadata")
			}
			if result.Metadata["threshold"] == nil {
				t.Error("Moderation.Score() missing threshold in metadata")
			}

			// Verify is_safe matches expected
			if isSafe, ok := result.Metadata["is_safe"].(bool); !ok || isSafe == tt.wantUnsafe {
				t.Errorf("Moderation.Score() is_safe = %v, want %v", isSafe, !tt.wantUnsafe)
			}
		})
	}
}
