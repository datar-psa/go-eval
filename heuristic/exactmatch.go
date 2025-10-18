package heuristic

import (
	"context"
	"strings"

	goeval "github.com/datar-psa/go-eval"
)

// ExactMatchOptions configures the ExactMatch scorer
type ExactMatchOptions struct {
	// CaseInsensitive determines if the comparison should ignore case
	CaseInsensitive bool
	// TrimWhitespace determines if leading and trailing whitespace should be trimmed
	TrimWhitespace bool
}

// ExactMatch returns a scorer that checks if the output exactly matches the expected value
func ExactMatch(opts ExactMatchOptions) goeval.Scorer {
	return &exactMatchScorer{opts: opts}
}

type exactMatchScorer struct {
	opts ExactMatchOptions
}

func (s *exactMatchScorer) Score(ctx context.Context, in goeval.ScoreInputs) goeval.Score {
	result := goeval.Score{
		Name:     "ExactMatch",
		Metadata: make(map[string]any),
	}

	if in.Expected == "" {
		result.Error = goeval.ErrNoExpectedValue
		result.Score = 0
		return result
	}

	outputToCompare := in.Output
	expectedToCompare := in.Expected

	if s.opts.TrimWhitespace {
		outputToCompare = strings.TrimSpace(outputToCompare)
		expectedToCompare = strings.TrimSpace(expectedToCompare)
	}

	if s.opts.CaseInsensitive {
		outputToCompare = strings.ToLower(outputToCompare)
		expectedToCompare = strings.ToLower(expectedToCompare)
	}

	if outputToCompare == expectedToCompare {
		result.Score = 1.0
	} else {
		result.Score = 0.0
	}

	result.Metadata["case_insensitive"] = s.opts.CaseInsensitive
	result.Metadata["trim_whitespace"] = s.opts.TrimWhitespace
	result.Metadata["output_length"] = len(in.Output)
	result.Metadata["expected_length"] = len(in.Expected)

	return result
}
