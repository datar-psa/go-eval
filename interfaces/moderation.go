package interfaces

import "context"

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
