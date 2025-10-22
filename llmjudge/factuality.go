package llmjudge

import (
	"context"
	"fmt"

	"github.com/datar-psa/goeval/api"
)

// FactualityOptions configures the Factuality scorer
type FactualityOptions struct {
	// Additional configuration options can be added here
}

// Factuality returns a scorer that uses an LLM to evaluate if the output is factually consistent with the expected answer
// This scorer uses chain-of-thought reasoning to determine factuality
func Factuality(llm api.LLMGenerator, opts FactualityOptions) api.Scorer {
	return &factualityScorer{
		opts: opts,
		llm:  llm,
	}
}

type factualityScorer struct {
	opts FactualityOptions
	llm  api.LLMGenerator
}

const factualityPromptTemplate = `You are comparing a submitted answer to an expert answer on a given question. Here is the data:
[BEGIN DATA]
************
[Question]: %s
************
[Expert]: %s
************
[Submission]: %s
************
[END DATA]

Compare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.
The submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. Answer the question by selecting one of the following options:
(A) The submitted answer contains all the same details as the expert answer (EXCELLENT).
(B) The answers differ, but these differences don't matter from the perspective of factuality (VERY GOOD).
(C) The submitted answer is a superset of the expert answer and is fully consistent with it (GOOD).
(D) The submitted answer is a subset of the expert answer and is fully consistent with it (FAIR).
(E) There is a disagreement between the submitted answer and the expert answer (POOR).

Provide your assessment with a choice (A, B, C, D, or E) and a detailed explanation of your reasoning.`

func (s *factualityScorer) Score(ctx context.Context, in api.ScoreInputs) api.Score {
	result := api.Score{
		Name:     "Factuality",
		Metadata: make(map[string]any),
	}

	if in.Expected == "" {
		result.Error = api.ErrNoExpectedValue
		result.Score = 0
		return result
	}

	if s.llm == nil {
		result.Error = fmt.Errorf("LLM generator is required")
		result.Score = 0
		return result
	}

	prompt := fmt.Sprintf(factualityPromptTemplate, in.Input, in.Expected, in.Output)

	// Define schema for structured response
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"choice": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"A", "B", "C", "D", "E"},
				"description": "The factuality assessment choice: (A) same details (EXCELLENT), (B) differences don't matter (VERY GOOD), (C) superset and consistent (GOOD), (D) subset and consistent (FAIR), (E) disagreement (POOR)",
			},
			"explanation": map[string]interface{}{
				"type":        "string",
				"description": "Detailed explanation of the factuality assessment",
			},
		},
		"required": []string{"choice", "explanation"},
	}

	// Use StructuredGenerate to get structured response
	structuredResponse, err := s.llm.StructuredGenerate(ctx, prompt, schema)
	if err != nil {
		result.Error = fmt.Errorf("LLM generation failed: %v", err)
		result.Score = 0
		return result
	}

	// Extract choice and explanation from structured response
	choice, ok := structuredResponse["choice"].(string)
	if !ok {
		result.Error = fmt.Errorf("failed to extract choice from structured response")
		result.Score = 0
		result.Metadata["raw_response"] = structuredResponse
		return result
	}

	explanation, ok := structuredResponse["explanation"].(string)
	if !ok {
		result.Error = fmt.Errorf("failed to extract explanation from structured response")
		result.Score = 0
		result.Metadata["raw_response"] = structuredResponse
		return result
	}

	// Map choice to score using school-style grading (A=best, E=worst)
	choiceScores := map[string]float64{
		"A": 1.0, // same details
		"B": 0.8, // differences don't matter
		"C": 0.6, // superset and consistent
		"D": 0.4, // subset and consistent
		"E": 0.0, // disagreement
	}

	result.Score = choiceScores[choice]
	result.Metadata["choice"] = choice
	result.Metadata["explanation"] = explanation
	result.Metadata["raw_response"] = structuredResponse

	return result
}
