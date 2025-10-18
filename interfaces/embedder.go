package interfaces

import "context"

// Embedder generates vector embeddings for text
type Embedder interface {
	// Embed generates an embedding vector for the given text
	// Returns a normalized vector (length = 1) suitable for cosine similarity
	Embed(ctx context.Context, text string) ([]float64, error)
}
