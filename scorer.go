package goeval

import "context"

// Score represents the result of an evaluation
type Score struct {
	// Name identifies the scorer that produced this result
	Name string
	// Score is a value between 0 and 1, where 1 is the best possible score
	Score float64
	// Metadata contains additional information about the scoring process
	Metadata map[string]any
	// Error contains any error that occurred during scoring
	Error error
}

// Scorer evaluates the quality of an output
type Scorer interface {
	// Score evaluates the output and returns a score
	// input: the input provided to the model
	// output: the actual output from the model
	// expected: the expected/reference output (optional, can be empty)
	Score(ctx context.Context, input, output, expected string) Score
}
