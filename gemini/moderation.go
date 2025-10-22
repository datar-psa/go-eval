package gemini

import (
	"context"
	"fmt"

	language "cloud.google.com/go/language/apiv1"
	languagepb "cloud.google.com/go/language/apiv1/languagepb"
	"github.com/datar-psa/goeval/api"
)

// GoogleLanguageProvider implements ModerationProvider using Google Cloud Natural Language API client
type GoogleLanguageProvider struct {
	client *language.Client
}

// NewGoogleLanguageProvider creates a new provider using a preconfigured *language.Client (auth handled by caller)
func NewGoogleLanguageProvider(client *language.Client) api.ModerationProvider {
	return &GoogleLanguageProvider{client: client}
}

// Moderate analyzes content for safety using Google Cloud Natural Language API
func (p *GoogleLanguageProvider) Moderate(ctx context.Context, content string) (*api.ModerationResult, error) {
	if p.client == nil {
		return nil, fmt.Errorf("language client is required")
	}

	req := &languagepb.ModerateTextRequest{
		Document: &languagepb.Document{
			Type: languagepb.Document_PLAIN_TEXT,
			Source: &languagepb.Document_Content{
				Content: content,
			},
		},
	}

	resp, err := p.client.ModerateText(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("moderate text failed: %w", err)
	}

	categories := make([]api.ModerationCategory, 0, len(resp.ModerationCategories))
	for _, c := range resp.ModerationCategories {
		categories = append(categories, api.ModerationCategory{
			Name:       mapCategoryName(c.Name),
			Confidence: float64(c.Confidence),
		})
	}

	return &api.ModerationResult{Categories: categories}, nil
}

// mapCategoryName maps Google Cloud Natural Language API category names to developer-friendly names
func mapCategoryName(googleCategory string) string {
	switch googleCategory {
	case "Toxic":
		return "Toxic"
	case "Derogatory":
		return "Derogatory"
	case "Violent":
		return "Violent"
	case "Sexual":
		return "Sexual"
	case "Insult":
		return "Insult"
	case "Profanity":
		return "Profanity"
	case "Death, Harm & Tragedy":
		return "DeathHarmTragedy"
	case "Firearms & Weapons":
		return "FirearmsWeapons"
	case "Public Safety":
		return "PublicSafety"
	case "Health":
		return "Health"
	case "Religion & Belief":
		return "ReligionBelief"
	case "Illicit Drugs":
		return "IllicitDrugs"
	case "War & Conflict":
		return "WarConflict"
	case "Finance":
		return "Finance"
	case "Politics":
		return "Politics"
	case "Legal":
		return "Legal"
	default:
		// Return original name if not recognized
		return googleCategory
	}
}
