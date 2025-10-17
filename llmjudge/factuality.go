package llmjudge

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	goeval "github.com/datar-psa/go-eval"
)

// FactualityOptions configures the Factuality scorer
type FactualityOptions struct {
	// LLM is the language model generator to use for evaluation
	LLM goeval.LLMGenerator
}

// Factuality returns a scorer that uses an LLM to evaluate if the output is factually consistent with the expected answer
// This scorer uses chain-of-thought reasoning to determine factuality
func Factuality(opts FactualityOptions) goeval.Scorer {
	return &factualityScorer{opts: opts}
}

type factualityScorer struct {
	opts FactualityOptions
}

const factualityPromptTemplate = `You are evaluating the factual accuracy of an AI assistant's answer.

Input: %s
Expected Answer: %s
Actual Output: %s

Please evaluate if the actual output is factually consistent with the expected answer.
Consider the output correct if it conveys the same core facts, even if wording differs.

Think step by step:
1. Identify the key facts in the expected answer
2. Check if these facts are present in the actual output
3. Check if there are any contradicting facts in the actual output

Then provide your final answer as a score from 0 to 10, where:
- 0 = completely wrong or contradictory
- 5 = partially correct
- 10 = fully correct and factually consistent

End your response with: "SCORE: X" where X is a number from 0 to 10.`

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

	// Extract score from response
	score, reasoning, err := extractScore(response)
	if err != nil {
		result.Error = fmt.Errorf("failed to extract score: %w", err)
		result.Score = 0
		result.Metadata["raw_response"] = response
		return result
	}

	// Normalize score from 0-10 to 0-1
	result.Score = float64(score) / 10.0
	result.Metadata["raw_score"] = score
	result.Metadata["reasoning"] = reasoning
	result.Metadata["raw_response"] = response

	return result
}

// extractScore extracts the score from the LLM response
// Returns the score (0-10), reasoning, and any error
func extractScore(response string) (int, string, error) {
	// Look for "SCORE: X" pattern
	scoreRegex := regexp.MustCompile(`SCORE:\s*(\d+)`)
	matches := scoreRegex.FindStringSubmatch(response)

	if len(matches) < 2 {
		return 0, "", fmt.Errorf("could not find SCORE pattern in response")
	}

	score, err := strconv.Atoi(matches[1])
	if err != nil {
		return 0, "", fmt.Errorf("invalid score value: %w", err)
	}

	if score < 0 || score > 10 {
		return 0, "", fmt.Errorf("score out of range: %d", score)
	}

	// Extract reasoning (everything before the SCORE line)
	scoreLine := matches[0]
	scoreIndex := strings.Index(response, scoreLine)
	reasoning := strings.TrimSpace(response[:scoreIndex])

	return score, reasoning, nil
}
