package llmjudge

import (
	"context"
	"fmt"

	goeval "github.com/datar-psa/go-eval"
	"github.com/datar-psa/go-eval/interfaces"
)

// TonalityOptions configures the Tonality scorer
type TonalityOptions struct {
	// Individual weights; if all are 0, defaults to equal weights
	ProfessionalismWeight float64
	KindnessWeight        float64
	ClarityWeight         float64
	HelpfulnessWeight     float64

	// Threshold: if any used category (non-zero weight) is below this threshold, score becomes 0
	// Range: 0.0-1.0, where 0.0 means no threshold (default)
	Threshold float64
}

// Tonality returns a scorer that evaluates professionalism, kindness, clarity, and helpfulness
// in a single LLM-judge call using anchored A–E categories.
// The final score is a weighted blend of the dimensions, normalized to [0,1].
func Tonality(llm interfaces.LLMGenerator, opts TonalityOptions) goeval.Scorer {
	return &tonalityScorer{
		opts: opts,
		llm:  llm,
	}
}

type tonalityScorer struct {
	opts TonalityOptions
	llm  interfaces.LLMGenerator
}

const tonalityPromptTemplate = `You are evaluating the quality of an AI response across multiple dimensions. Be deterministic and concise.

[BEGIN DATA]
[Context]: %s
[Response]: %s
[END DATA]

Dimension anchors (use these precise anchors, not your own):
- Professionalism:
  A: casual/slang, confrontational, imprecise; chaotic formatting
  B: frequent informality; repeated imprecision
  C: generally professional; minor informality/sloppiness
  D: consistently professional; precise and neutral
  E: highly professional; precise, neutral, impeccably formatted
- Kindness:
  A: hostile, shaming, dismissive
  B: occasionally harsh/blaming
  C: neutral/polite
  D: empathetic, supportive
  E: exemplary empathy and care
- Clarity:
  A: hard to understand; disorganized
  B: somewhat unclear; weak structure
  C: understandable; some redundancy
  D: clear, well-structured
  E: exceptionally clear; concise and well structured
- Helpfulness:
  A: off-topic; no actionable guidance
  B: partially relevant; little actionability
  C: addresses request; limited actionability
  D: directly addresses request; actionable steps
  E: fully addresses request; step-by-step; anticipates edge cases

Instructions:
- Rate each dimension independently with one of A, B, C, D, E.
- For each dimension, provide: confidence (0.0–1.0), a short explanation (<=30 words), and 1–3 short quotes from the Response as evidence.
`

func (s *tonalityScorer) Score(ctx context.Context, in goeval.ScoreInputs) goeval.Score {
	result := goeval.Score{
		Name:     "Tonality",
		Metadata: make(map[string]any),
	}

	if s.llm == nil {
		result.Error = fmt.Errorf("LLM generator is required")
		result.Score = 0
		return result
	}

	prompt := fmt.Sprintf(tonalityPromptTemplate, in.Input, in.Output)

	// Define schema for structured response
	schema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"professionalism": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"A", "B", "C", "D", "E"},
				"description": "Professionalism rating (A–E) with anchored definitions",
			},
			"kindness": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"A", "B", "C", "D", "E"},
				"description": "Kindness rating (A–E) with anchored definitions",
			},
			"clarity": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"A", "B", "C", "D", "E"},
				"description": "Clarity rating (A–E) with anchored definitions",
			},
			"helpfulness": map[string]interface{}{
				"type":        "string",
				"enum":        []string{"A", "B", "C", "D", "E"},
				"description": "Helpfulness rating (A–E) with anchored definitions",
			},

			// Optional confidences (0.0–1.0)
			"professionalism_confidence": map[string]interface{}{"type": "number"},
			"kindness_confidence":        map[string]interface{}{"type": "number"},
			"clarity_confidence":         map[string]interface{}{"type": "number"},
			"helpfulness_confidence":     map[string]interface{}{"type": "number"},

			// Optional explanations
			"professionalism_explanation": map[string]interface{}{"type": "string"},
			"kindness_explanation":        map[string]interface{}{"type": "string"},
			"clarity_explanation":         map[string]interface{}{"type": "string"},
			"helpfulness_explanation":     map[string]interface{}{"type": "string"},

			// Optional evidence arrays (quotes)
			"professionalism_evidence": map[string]interface{}{
				"type":  "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"kindness_evidence": map[string]interface{}{
				"type":  "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"clarity_evidence": map[string]interface{}{
				"type":  "array",
				"items": map[string]interface{}{"type": "string"},
			},
			"helpfulness_evidence": map[string]interface{}{
				"type":  "array",
				"items": map[string]interface{}{"type": "string"},
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

	// Base LLM scores per dimension
	baseScores := [4]float64{
		choiceToScore[choices[0]], // professionalism
		choiceToScore[choices[1]], // kindness
		choiceToScore[choices[2]], // clarity
		choiceToScore[choices[3]], // helpfulness
	}

	// Optional confidences (not used for scoring; surfaced in metadata)
	confidences := [4]float64{0.7, 0.7, 0.7, 0.7}
	if v, ok := structuredResponse["professionalism_confidence"].(float64); ok {
		confidences[0] = clamp01(v)
	}
	if v, ok := structuredResponse["kindness_confidence"].(float64); ok {
		confidences[1] = clamp01(v)
	}
	if v, ok := structuredResponse["clarity_confidence"].(float64); ok {
		confidences[2] = clamp01(v)
	}
	if v, ok := structuredResponse["helpfulness_confidence"].(float64); ok {
		confidences[3] = clamp01(v)
	}

	// Build weights array from individual fields; if weight is 0, that dimension is not relevant
	weights := [4]float64{
		s.opts.ProfessionalismWeight,
		s.opts.KindnessWeight,
		s.opts.ClarityWeight,
		s.opts.HelpfulnessWeight,
	}

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

	// Calculate weighted score using base LLM scores
	finalScore := 0.0
	for i := 0; i < 4; i++ {
		finalScore += weights[i] * baseScores[i]
	}

	// Apply threshold: if any used category (non-zero weight) is below threshold, score becomes 0
	if s.opts.Threshold > 0 {
		for i := 0; i < 4; i++ {
			if weights[i] > 0 && baseScores[i] < s.opts.Threshold {
				finalScore = 0.0
				break
			}
		}
	}

	result.Score = finalScore
	result.Metadata["professionalism.choice"] = choices[0]
	result.Metadata["professionalism.score"] = baseScores[0]
	result.Metadata["professionalism.confidence"] = confidences[0]
	result.Metadata["kindness.choice"] = choices[1]
	result.Metadata["kindness.score"] = baseScores[1]
	result.Metadata["kindness.confidence"] = confidences[1]
	result.Metadata["clarity.choice"] = choices[2]
	result.Metadata["clarity.score"] = baseScores[2]
	result.Metadata["clarity.confidence"] = confidences[2]
	result.Metadata["helpfulness.choice"] = choices[3]
	result.Metadata["helpfulness.score"] = baseScores[3]
	result.Metadata["helpfulness.confidence"] = confidences[3]
	result.Metadata["weights.professionalism"] = weights[0]
	result.Metadata["weights.kindness"] = weights[1]
	result.Metadata["weights.clarity"] = weights[2]
	result.Metadata["weights.helpfulness"] = weights[3]
	result.Metadata["threshold"] = s.opts.Threshold
	result.Metadata["raw_response"] = structuredResponse

	return result
}

// returnError is a helper function to set error metadata consistently
func (s *tonalityScorer) returnError(result *goeval.Score, err error, rawResponse interface{}) goeval.Score {
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

// --- Helpers ---

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// (no other helpers)
