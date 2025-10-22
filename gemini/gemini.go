package gemini

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/datar-psa/goeval/api"
	"google.golang.org/genai"
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

// StructuredGenerate implements LLMGenerator.StructuredGenerate
func (g *Generator) StructuredGenerate(ctx context.Context, prompt string, schema map[string]interface{}) (map[string]interface{}, error) {
	// Convert schema to genai.Schema
	genaiSchema, err := g.convertToGenaiSchema(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to convert schema: %w", err)
	}

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
		&genai.GenerateContentConfig{
			ResponseMIMEType: "application/json",
			ResponseSchema:   genaiSchema,
		},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to generate content: %w", err)
	}

	if len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates returned")
	}

	if len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("no parts in response")
	}

	responseText := resp.Candidates[0].Content.Parts[0].Text

	// Parse the JSON response
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(responseText), &result); err != nil {
		return nil, fmt.Errorf("failed to parse JSON response: %w, response: %s", err, responseText)
	}

	return result, nil
}

// convertToGenaiSchema converts a map[string]interface{} schema to genai.Schema
func (g *Generator) convertToGenaiSchema(schema map[string]interface{}) (*genai.Schema, error) {
	// Convert to JSON first, then unmarshal into genai.Schema
	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	var genaiSchema genai.Schema
	if err := json.Unmarshal(schemaJSON, &genaiSchema); err != nil {
		return nil, fmt.Errorf("failed to unmarshal schema: %w", err)
	}

	return &genaiSchema, nil
}

// Verify that Generator implements LLMGenerator
var _ api.LLMGenerator = (*Generator)(nil)
