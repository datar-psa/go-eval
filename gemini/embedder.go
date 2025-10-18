package gemini

import (
	"context"
	"fmt"

	"github.com/datar-psa/go-eval/interfaces"
	"google.golang.org/genai"
)

// Embedder wraps a genai.Client to implement the Embedder interface
type Embedder struct {
	client    *genai.Client
	modelName string
}

// NewEmbedder creates a new Gemini embedder
// client: genai.Client from google.golang.org/genai
// modelName: the embedding model to use (e.g., "text-embedding-005")
func NewEmbedder(client *genai.Client, modelName string) *Embedder {
	return &Embedder{
		client:    client,
		modelName: modelName,
	}
}

// Embed implements Embedder.Embed
// Note: This uses the Embedding API which is separate from the text generation API
func (e *Embedder) Embed(ctx context.Context, text string) ([]float64, error) {
	// Prepare content for embedding
	contents := []*genai.Content{
		{
			Parts: []*genai.Part{
				{Text: text},
			},
		},
	}

	// Call the embeddings endpoint using the EmbedContent method
	result, err := e.client.Models.EmbedContent(ctx, e.modelName, contents, &genai.EmbedContentConfig{})
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	if len(result.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	if len(result.Embeddings[0].Values) == 0 {
		return nil, fmt.Errorf("empty embedding vector")
	}

	// Convert []float32 to []float64
	values := result.Embeddings[0].Values
	embedding := make([]float64, len(values))
	for i, v := range values {
		embedding[i] = float64(v)
	}

	return embedding, nil
}

// Verify that Embedder implements interfaces.Embedder
var _ interfaces.Embedder = (*Embedder)(nil)
