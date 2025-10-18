package interfaces

import "context"

// LLMGenerator is an interface for generating text using an LLM
// This interface must be implemented by library consumers
// A Gemini implementation is provided in the gemini subpackage
type LLMGenerator interface {
	// Generate generates text based on the provided prompt
	// Returns the generated text or an error
	Generate(ctx context.Context, prompt string) (string, error)
}
