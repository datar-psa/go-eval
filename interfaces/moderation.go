package interfaces

import "context"

// ModerationCategory represents a safety category with confidence score
type ModerationCategory struct {
	Name       string  `json:"name"`
	Confidence float64 `json:"confidence"`
}

// ModerationResult represents the result of content moderation
type ModerationResult struct {
	Categories    []ModerationCategory `json:"categories"`
	IsSafe        bool                 `json:"is_safe"`
	MaxConfidence float64              `json:"max_confidence"`
}

// ModerationProvider is an interface for content moderation
// This interface must be implemented by library consumers
// A Google Cloud Natural Language implementation is provided in the moderation subpackage
type ModerationProvider interface {
	// Moderate analyzes content for safety and returns moderation results
	// Returns the moderation result or an error
	Moderate(ctx context.Context, content string) (*ModerationResult, error)
}
