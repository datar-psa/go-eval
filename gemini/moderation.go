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
		ModerationCategories []interfaces.ModerationCategory `json:"moderationCategories"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&apiResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Process results
	maxConfidence := 0.0
	isSafe := true

	for _, category := range apiResponse.ModerationCategories {
		if category.Confidence > maxConfidence {
			maxConfidence = category.Confidence
		}
		// Consider content unsafe if any category has high confidence
		if category.Confidence > 0.5 {
			isSafe = false
		}
	}

	return &interfaces.ModerationResult{
		Categories:    apiResponse.ModerationCategories,
		IsSafe:        isSafe,
		MaxConfidence: maxConfidence,
	}, nil
}
