package llmjudge

import (
	"context"
	"fmt"

	goeval "github.com/datar-psa/go-eval"
	"github.com/datar-psa/go-eval/interfaces"
)

// ToneRubricOptions configures the ToneRubric scorer
type ToneRubricOptions struct {
	// Weights for each dimension in order: [professionalism, kindness, clarity, helpfulness]
	// If weight is 0, that dimension is excluded from scoring
	// If all weights are 0, defaults to equal weights
	Weights [4]float64
}

// ToneRubric returns a scorer that evaluates professionalism, kindness, clarity, and helpfulness
// in a single LLM-judge call using a rubric with A–E categories.
// The final score is a weighted blend of the dimensions, normalized to [0,1].
func ToneRubric(llm interfaces.LLMGenerator, opts ToneRubricOptions) goeval.Scorer {
	return &toneRubricScorer{
		opts: opts,
		llm:  llm,
	}
}

type toneRubricScorer struct {
	opts ToneRubricOptions
	llm  interfaces.LLMGenerator
}

const toneRubricPromptTemplate = `You are evaluating the quality of an AI response across multiple dimensions.

[BEGIN DATA]
[Context]: %s
[Response]: %s
[END DATA]

Definitions:
- Professionalism: clarity, formality, appropriateness to context, respectful language, precision; avoids slang/exaggeration and personal attacks.
- Kindness: empathy, warmth, supportive and non-judgmental language; avoids blame or harshness.
- Clarity: clear, concise, easy to understand; well-structured and organized.
- Helpfulness: provides useful, actionable information; addresses the user's needs effectively.

Rate each dimension independently using these categories:
(A) Unacceptable
(B) Below acceptable
(C) Adequate
(D) Good
(E) Excellent

Provide your assessment with ratings for each dimension.`

func (s *toneRubricScorer) Score(ctx context.Context, in goeval.ScoreInputs) goeval.Score {
	result := goeval.Score{
		Name:     "ToneRubric",
		Metadata: make(map[string]any),
	}

	if s.llm == nil {
		result.Error = fmt.Errorf("LLM generator is required")
		result.Score = 0
		return result
	}

	prompt := fmt.Sprintf(toneRubricPromptTemplate, in.Input, in.Output)

	// Define schema for structured response
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"professionalism": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"A", "B", "C", "D", "E"},
				"description": "Professionalism rating",
			},
			"kindness": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"A", "B", "C", "D", "E"},
				"description": "Kindness rating",
			},
			"clarity": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"A", "B", "C", "D", "E"},
				"description": "Clarity rating",
			},
			"helpfulness": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"A", "B", "C", "D", "E"},
				"description": "Helpfulness rating",
			},
		},
		"required": []string{"professionalism", "kindness", "clarity", "helpfulness"},
	}

	// Use StructuredGenerate to get structured response
	structuredResponse, err := s.llm.StructuredGenerate(ctx, prompt, schema)
	if err != nil {
		return s.returnError(&result, fmt.Errorf("%w: %v", goeval.ErrLLMGenerationFailed, err), nil)
	}

	// Extract choices from structured response
	choices := [4]string{}

	profChoice, ok := structuredResponse["professionalism"].(string)
	if !ok {
		return s.returnError(&result, fmt.Errorf("failed to extract professionalism choice from structured response"), structuredResponse)
	}
	choices[0] = profChoice

	kindChoice, ok := structuredResponse["kindness"].(string)
	if !ok {
		return s.returnError(&result, fmt.Errorf("failed to extract kindness choice from structured response"), structuredResponse)
	}
	choices[1] = kindChoice

	clarityChoice, ok := structuredResponse["clarity"].(string)
	if !ok {
		return s.returnError(&result, fmt.Errorf("failed to extract clarity choice from structured response"), structuredResponse)
	}
	choices[2] = clarityChoice

	helpfulnessChoice, ok := structuredResponse["helpfulness"].(string)
	if !ok {
		return s.returnError(&result, fmt.Errorf("failed to extract helpfulness choice from structured response"), structuredResponse)
	}
	choices[3] = helpfulnessChoice

	// Map A–E to [0,1]
	choiceToScore := map[string]float64{
		"A": 0.0,
		"B": 0.25,
		"C": 0.5,
		"D": 0.75,
		"E": 1.0,
	}

	scores := [4]float64{
		choiceToScore[choices[0]], // professionalism
		choiceToScore[choices[1]], // kindness
		choiceToScore[choices[2]], // clarity
		choiceToScore[choices[3]], // helpfulness
	}

	// Process weights: if weight is 0, that dimension is not relevant
	weights := s.opts.Weights

	// Count non-zero weights
	nonZeroCount := 0
	for _, w := range weights {
		if w > 0 {
			nonZeroCount++
		}
	}

	// If all weights are 0 or negative, default to equal weights
	if nonZeroCount == 0 {
		for i := range weights {
			weights[i] = 0.25 // Equal weight for all 4 dimensions
		}
	} else {
		// Normalize weights to sum to 1
		sum := 0.0
		for _, w := range weights {
			if w > 0 {
				sum += w
			}
		}
		if sum > 0 {
			for i := range weights {
				if weights[i] > 0 {
					weights[i] /= sum
				}
			}
		}
	}

	// Calculate weighted score
	finalScore := 0.0
	for i, score := range scores {
		finalScore += weights[i] * score
	}

	result.Score = finalScore
	result.Metadata["professionalism.choice"] = choices[0]
	result.Metadata["professionalism.score"] = scores[0]
	result.Metadata["kindness.choice"] = choices[1]
	result.Metadata["kindness.score"] = scores[1]
	result.Metadata["clarity.choice"] = choices[2]
	result.Metadata["clarity.score"] = scores[2]
	result.Metadata["helpfulness.choice"] = choices[3]
	result.Metadata["helpfulness.score"] = scores[3]
	result.Metadata["weights.professionalism"] = weights[0]
	result.Metadata["weights.kindness"] = weights[1]
	result.Metadata["weights.clarity"] = weights[2]
	result.Metadata["weights.helpfulness"] = weights[3]
	result.Metadata["raw_response"] = structuredResponse

	return result
}

// returnError is a helper function to set error metadata consistently
func (s *toneRubricScorer) returnError(result *goeval.Score, err error, rawResponse interface{}) goeval.Score {
	result.Error = err
	result.Score = 0
	result.Metadata["raw_response"] = rawResponse
	result.Metadata["professionalism.choice"] = ""
	result.Metadata["professionalism.score"] = 0.0
	result.Metadata["kindness.choice"] = ""
	result.Metadata["kindness.score"] = 0.0
	result.Metadata["clarity.choice"] = ""
	result.Metadata["clarity.score"] = 0.0
	result.Metadata["helpfulness.choice"] = ""
	result.Metadata["helpfulness.score"] = 0.0
	result.Metadata["weights.professionalism"] = 0.0
	result.Metadata["weights.kindness"] = 0.0
	result.Metadata["weights.clarity"] = 0.0
	result.Metadata["weights.helpfulness"] = 0.0
	return *result
}
