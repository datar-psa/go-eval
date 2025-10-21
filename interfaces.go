package goeval

import "context"

// LLMGenerator is an interface for generating text using an LLM
// This interface must be implemented by library consumers
// A Gemini implementation is provided in the gemini subpackage
type LLMGenerator interface {
	// Generate generates text based on the provided prompt
	// Returns the generated text or an error
	Generate(ctx context.Context, prompt string) (string, error)

	// StructuredGenerate generates structured data based on the provided prompt and JSON schema
	// schema must be a valid JSON schema (map[string]interface{})
	// Returns the generated data as a map[string]interface{} or an error
	StructuredGenerate(ctx context.Context, prompt string, schema map[string]interface{}) (map[string]interface{}, error)
}

// Embedder generates vector embeddings for text
type Embedder interface {
	// Embed generates an embedding vector for the given text
	// Returns a normalized vector (length = 1) suitable for cosine similarity
	Embed(ctx context.Context, text string) ([]float64, error)
}

// ModerationCategories contains all supported moderation category names
// These are developer-friendly names that map to Google Cloud Natural Language API categories
var ModerationCategories []string = []string{
	"Toxic",
	"Derogatory",
	"Violent",
	"Sexual",
	"Insult",
	"Profanity",
	"DeathHarmTragedy",
	"FirearmsWeapons",
	"PublicSafety",
	"Health",
	"ReligionBelief",
	"IllicitDrugs",
	"WarConflict",
	"Finance",
	"Politics",
	"Legal",
}

// ModerationCategory represents a safety category with confidence score
type ModerationCategory struct {
	Name       string  `json:"name"`
	Confidence float64 `json:"confidence"`
}

// ModerationResult represents the result of content moderation
type ModerationResult struct {
	Categories []ModerationCategory `json:"categories"`
}

// ModerationProvider is an interface for content moderation
// This interface must be implemented by library consumers
// A Google Cloud Natural Language implementation is provided in the moderation subpackage
type ModerationProvider interface {
	// Moderate analyzes content for safety and returns moderation results
	// Returns the moderation result or an error
	Moderate(ctx context.Context, content string) (*ModerationResult, error)
}
