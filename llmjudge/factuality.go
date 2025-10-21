package llmjudge

import (
	"context"
	"fmt"

	goeval "github.com/datar-psa/go-eval"
	"github.com/datar-psa/go-eval/interfaces"
)

// FactualityOptions configures the Factuality scorer
type FactualityOptions struct {
	// Additional configuration options can be added here
}

// Factuality returns a scorer that uses an LLM to evaluate if the output is factually consistent with the expected answer
// This scorer uses chain-of-thought reasoning to determine factuality
func Factuality(llm interfaces.LLMGenerator, opts FactualityOptions) goeval.Scorer {
	return &factualityScorer{
		opts: opts,
		llm:  llm,
	}
}

type factualityScorer struct {
	opts FactualityOptions
	llm  interfaces.LLMGenerator
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
(A) The submitted answer is a subset of the expert answer and is fully consistent with it.
(B) The submitted answer is a superset of the expert answer and is fully consistent with it.
(C) The submitted answer contains all the same details as the expert answer.
(D) There is a disagreement between the submitted answer and the expert answer.
(E) The answers differ, but these differences don't matter from the perspective of factuality.

Provide your assessment with a choice (A, B, C, D, or E) and a detailed explanation of your reasoning.`

func (s *factualityScorer) Score(ctx context.Context, in goeval.ScoreInputs) goeval.Score {
	result := goeval.Score{
		Name:     "Factuality",
		Metadata: make(map[string]any),
	}

	if in.Expected == "" {
		result.Error = goeval.ErrNoExpectedValue
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
				"description": "The factuality assessment choice: (A) subset and consistent, (B) superset and consistent, (C) same details, (D) disagreement, (E) differences don't matter",
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
		result.Error = fmt.Errorf("%w: %v", goeval.ErrLLMGenerationFailed, err)
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

	// Map choice to score based on Braintrust scoring
	choiceScores := map[string]float64{
		"A": 0.4, // subset and consistent
		"B": 0.6, // superset and consistent
		"C": 1.0, // same details
		"D": 0.0, // disagreement
		"E": 1.0, // differences don't matter
	}

	result.Score = choiceScores[choice]
	result.Metadata["choice"] = choice
	result.Metadata["explanation"] = explanation
	result.Metadata["raw_response"] = structuredResponse

	return result
}
