package llmjudge

import (
	"context"
	"fmt"
	"regexp"

	goeval "github.com/datar-psa/go-eval"
	"github.com/datar-psa/go-eval/interfaces"
)

// FactualityOptions configures the Factuality scorer
type FactualityOptions struct {
	// LLM is the language model generator to use for evaluation
	LLM interfaces.LLMGenerator
}

// Factuality returns a scorer that uses an LLM to evaluate if the output is factually consistent with the expected answer
// This scorer uses chain-of-thought reasoning to determine factuality
func Factuality(opts FactualityOptions) goeval.Scorer {
	return &factualityScorer{opts: opts}
}

type factualityScorer struct {
	opts FactualityOptions
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

Answer with just the letter (A, B, C, D, or E).`

func (s *factualityScorer) Score(ctx context.Context, input, output, expected string) goeval.Score {
	result := goeval.Score{
		Name:     "Factuality",
		Metadata: make(map[string]any),
	}

	if expected == "" {
		result.Error = goeval.ErrNoExpectedValue
		result.Score = 0
		return result
	}

	if s.opts.LLM == nil {
		result.Error = fmt.Errorf("LLM generator is required")
		result.Score = 0
		return result
	}

	prompt := fmt.Sprintf(factualityPromptTemplate, input, expected, output)

	response, err := s.opts.LLM.Generate(ctx, prompt)
	if err != nil {
		result.Error = fmt.Errorf("%w: %v", goeval.ErrLLMGenerationFailed, err)
		result.Score = 0
		return result
	}

	// Extract choice from response
	choice, err := extractChoice(response)
	if err != nil {
		result.Error = fmt.Errorf("failed to extract choice: %w", err)
		result.Score = 0
		result.Metadata["raw_response"] = response
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
	result.Metadata["raw_response"] = response

	return result
}

// extractChoice extracts the choice from the LLM response
// Returns the choice (A, B, C, D, or E) and any error
func extractChoice(response string) (string, error) {
	// Look for single letter choices
	choiceRegex := regexp.MustCompile(`\b([ABCDE])\b`)
	matches := choiceRegex.FindStringSubmatch(response)

	if len(matches) < 2 {
		return "", fmt.Errorf("could not find valid choice (A, B, C, D, or E) in response")
	}

	choice := matches[1]
	return choice, nil
}
