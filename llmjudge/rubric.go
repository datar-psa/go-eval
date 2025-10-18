package llmjudge

import (
	"context"
	"fmt"
	"regexp"

	goeval "github.com/datar-psa/go-eval"
	"github.com/datar-psa/go-eval/interfaces"
)

// ToneRubricOptions configures the ToneRubric scorer
type ToneRubricOptions struct {
	// LLM is the language model generator used for evaluation
	LLM interfaces.LLMGenerator
	// Weights for each dimension in order: [professionalism, kindness, clarity, helpfulness]
	// If weight is 0, that dimension is excluded from scoring
	// If all weights are 0, defaults to equal weights
	Weights [4]float64
}

// ToneRubric returns a scorer that evaluates professionalism, kindness, clarity, and helpfulness
// in a single LLM-judge call using a rubric with A–E categories.
// The final score is a weighted blend of the dimensions, normalized to [0,1].
func ToneRubric(opts ToneRubricOptions) goeval.Scorer {
	return &toneRubricScorer{opts: opts}
}

type toneRubricScorer struct {
	opts ToneRubricOptions
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

Answer with exactly:
PROFESSIONALISM: <A|B|C|D|E>
KINDNESS: <A|B|C|D|E>
CLARITY: <A|B|C|D|E>
HELPFULNESS: <A|B|C|D|E>`

func (s *toneRubricScorer) Score(ctx context.Context, input, output, expected string) goeval.Score {
	result := goeval.Score{
		Name:     "ToneRubric",
		Metadata: make(map[string]any),
	}

	if s.opts.LLM == nil {
		result.Error = fmt.Errorf("LLM generator is required")
		result.Score = 0
		return result
	}

	prompt := fmt.Sprintf(toneRubricPromptTemplate, input, output)

	response, err := s.opts.LLM.Generate(ctx, prompt)
	if err != nil {
		result.Error = fmt.Errorf("%w: %v", goeval.ErrLLMGenerationFailed, err)
		result.Score = 0
		// Set metadata for error case
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
		return result
	}

	choices, err := extractToneChoices(response)
	if err != nil {
		result.Error = fmt.Errorf("failed to extract rubric choices: %w", err)
		result.Score = 0
		result.Metadata["raw_response"] = response
		// Set metadata for parsing error case
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
		return result
	}

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
	result.Metadata["raw_response"] = response

	return result
}

// extractToneChoices parses the LLM response for all four dimension letters
// Returns [professionalism, kindness, clarity, helpfulness] choices
func extractToneChoices(response string) ([4]string, error) {
	var choices [4]string

	profRe := regexp.MustCompile(`(?i)PROFESSIONALISM:\s*([ABCDE])`)
	kindRe := regexp.MustCompile(`(?i)KINDNESS:\s*([ABCDE])`)
	clarityRe := regexp.MustCompile(`(?i)CLARITY:\s*([ABCDE])`)
	helpfulnessRe := regexp.MustCompile(`(?i)HELPFULNESS:\s*([ABCDE])`)

	profMatches := profRe.FindStringSubmatch(response)
	kindMatches := kindRe.FindStringSubmatch(response)
	clarityMatches := clarityRe.FindStringSubmatch(response)
	helpfulnessMatches := helpfulnessRe.FindStringSubmatch(response)

	if len(profMatches) < 2 || len(kindMatches) < 2 || len(clarityMatches) < 2 || len(helpfulnessMatches) < 2 {
		return choices, fmt.Errorf("missing one or more dimension choices in response")
	}

	choices[0] = profMatches[1]        // professionalism
	choices[1] = kindMatches[1]        // kindness
	choices[2] = clarityMatches[1]     // clarity
	choices[3] = helpfulnessMatches[1] // helpfulness

	return choices, nil
}
