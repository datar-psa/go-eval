package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/datar-psa/go-eval/interfaces"
)

// GoogleCloudOptions configures the Google Cloud Natural Language moderation provider
type GoogleCloudOptions struct {
	// HTTPClient is used to make requests to Google Cloud Natural Language API
	HTTPClient *http.Client
	// ProjectID is the Google Cloud project ID
	ProjectID string
	// APIKey is the Google Cloud API key (alternative to service account)
	APIKey string
}

// GoogleCloudProvider implements ModerationProvider using Google Cloud Natural Language API
type GoogleCloudProvider struct {
	opts GoogleCloudOptions
}

// NewGoogleCloudProvider creates a new Google Cloud Natural Language moderation provider
func NewGoogleCloudProvider(opts GoogleCloudOptions) interfaces.ModerationProvider {
	return &GoogleCloudProvider{opts: opts}
}

// Moderate analyzes content for safety using Google Cloud Natural Language API
func (p *GoogleCloudProvider) Moderate(ctx context.Context, content string) (*interfaces.ModerationResult, error) {
	if p.opts.HTTPClient == nil {
		return nil, fmt.Errorf("HTTP client is required")
	}

	if p.opts.ProjectID == "" && p.opts.APIKey == "" {
		return nil, fmt.Errorf("either ProjectID or APIKey is required")
	}

	// Prepare request body
	requestBody := map[string]interface{}{
		"document": map[string]interface{}{
			"type":    "PLAIN_TEXT",
			"content": content,
		},
	}

	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	// Build URL
	var apiURL string
	if p.opts.APIKey != "" {
		apiURL = fmt.Sprintf("https://language.googleapis.com/v1/documents:moderateText?key=%s", url.QueryEscape(p.opts.APIKey))
	} else {
		apiURL = "https://language.googleapis.com/v1/documents:moderateText"
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, "POST", apiURL, strings.NewReader(string(jsonBody)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json; charset=utf-8")

	// Make request
	resp, err := p.opts.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status %d", resp.StatusCode)
	}

	// Parse response
	var apiResponse struct {
		ModerationCategories []struct {
			Name       string  `json:"name"`
			Confidence float64 `json:"confidence"`
		} `json:"moderationCategories"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&apiResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Map Google Cloud categories to standardized names
	categories := make([]interfaces.ModerationCategory, 0, len(apiResponse.ModerationCategories))

	for _, category := range apiResponse.ModerationCategories {
		// Map Google Cloud category names to developer-friendly names
		standardizedName := mapCategoryName(category.Name)

		categories = append(categories, interfaces.ModerationCategory{
			Name:       standardizedName,
			Confidence: category.Confidence,
		})
	}

	return &interfaces.ModerationResult{
		Categories: categories,
	}, nil
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
