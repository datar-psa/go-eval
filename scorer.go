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

// ScoreInputs carries inputs for scoring across different scorers.
//
// Fields usage conventions:
// - Output:   the actual output produced by the model (required for most scorers)
// - Expected: the reference/expected output (optional depending on scorer)
// - Input:    the original prompt/context/question given to the model (optional)
type ScoreInputs struct {
	Output   string
	Expected string
	Input    string
}

// Scorer evaluates the quality of an output
type Scorer interface {
	// Score evaluates the output and returns a score
	// in: container for output/expected/input depending on scorer needs
	Score(ctx context.Context, in ScoreInputs) Score
}
