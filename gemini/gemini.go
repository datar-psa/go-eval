package gemini

import (
	"context"
	"fmt"

	"google.golang.org/genai"

	goeval "github.com/datar-psa/go-eval"
)

// Generator wraps a genai.Client to implement the LLMGenerator interface
type Generator struct {
	client    *genai.Client
	modelName string
}

// NewGenerator creates a new Gemini generator
// client: genai.Client from google.golang.org/genai
// modelName: the model to use (e.g., "gemini-2.5-flash")
func NewGenerator(client *genai.Client, modelName string) *Generator {
	return &Generator{
		client:    client,
		modelName: modelName,
	}
}

// Generate implements LLMGenerator.Generate
func (g *Generator) Generate(ctx context.Context, prompt string) (string, error) {
	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: prompt},
		},
	}

	resp, err := g.client.Models.GenerateContent(
		ctx,
		g.modelName,
		[]*genai.Content{content},
		&genai.GenerateContentConfig{},
	)
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	if len(resp.Candidates) == 0 {
		return "", fmt.Errorf("no candidates returned")
	}

	if len(resp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("no parts in response")
	}

	return resp.Candidates[0].Content.Parts[0].Text, nil
}

// Verify that Generator implements LLMGenerator
var _ goeval.LLMGenerator = (*Generator)(nil)
