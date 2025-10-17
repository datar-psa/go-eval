package goeval

import "errors"

var (
	// ErrNoExpectedValue is returned when an expected value is required but not provided
	ErrNoExpectedValue = errors.New("expected value is required for this scorer")
	// ErrLLMGenerationFailed is returned when LLM generation fails
	ErrLLMGenerationFailed = errors.New("LLM generation failed")
)
