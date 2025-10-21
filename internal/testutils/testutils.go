package testutils

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/areknoster/hypert"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/genai"

	"github.com/datar-psa/goeval/gemini"
)

// ShouldUpdate returns true if tests should update cached HTTP responses
// Set UPDATE_TESTS=true environment variable to update cached responses
func ShouldUpdate() bool {
	return os.Getenv("UPDATE_TESTS") == "true"
}

// HypertClientConfig configures hypert client creation
type HypertClientConfig struct {
	TestDataDir string
	SubDir      string // Optional subdirectory for organizing test data
}

// NewHypertClient creates a new hypert client for caching HTTP requests
// This is useful for integration tests that make external API calls
func NewHypertClient(t *testing.T, config HypertClientConfig) *http.Client {
	testDataDir := config.TestDataDir
	if config.SubDir != "" {
		testDataDir = filepath.Join(testDataDir, config.SubDir)
	}

	namingScheme, err := hypert.NewContentHashNamingScheme(testDataDir)
	if err != nil {
		t.Fatalf("failed to create naming scheme: %v", err)
	}

	hypertClient := hypert.TestClient(t, ShouldUpdate(),
		hypert.WithNamingScheme(namingScheme),
		hypert.WithRequestValidator(hypert.ComposedRequestValidator(
			hypert.PathValidator(),
			hypert.QueryParamsValidator(),
			hypert.MethodValidator(),
		)),
	)

	// If we're in record mode, wrap with OAuth2 authentication
	if ShouldUpdate() {
		ctx := context.Background()
		creds, err := google.FindDefaultCredentials(ctx)
		if err != nil {
			t.Fatalf("failed to get default credentials: %v", err)
		}
		return oauth2.NewClient(context.WithValue(ctx, oauth2.HTTPClient, hypertClient), creds.TokenSource)
	}

	return hypertClient
}

// quotaProjectTransport wraps an http.RoundTripper to add quota project header
type quotaProjectTransport struct {
	base      http.RoundTripper
	projectID string
}

func (t *quotaProjectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// Add quota project header
	req.Header.Set("X-Goog-User-Project", t.projectID)
	return t.base.RoundTrip(req)
}

// NewAuthenticatedHypertClient creates a new hypert client with OAuth2 authentication and quota project
// This is useful for Google Cloud APIs that require quota project to be set
func NewAuthenticatedHypertClient(t *testing.T, config HypertClientConfig, projectID string) *http.Client {
	testDataDir := config.TestDataDir
	if config.SubDir != "" {
		testDataDir = filepath.Join(testDataDir, config.SubDir)
	}

	namingScheme, err := hypert.NewContentHashNamingScheme(testDataDir)
	if err != nil {
		t.Fatalf("failed to create naming scheme: %v", err)
	}

	hypertClient := hypert.TestClient(t, ShouldUpdate(),
		hypert.WithNamingScheme(namingScheme),
		hypert.WithRequestValidator(hypert.ComposedRequestValidator(
			hypert.PathValidator(),
			hypert.QueryParamsValidator(),
			hypert.MethodValidator(),
		)),
	)

	// If we're in record mode, wrap with OAuth2 authentication and set quota project
	if ShouldUpdate() {
		ctx := context.Background()
		creds, err := google.FindDefaultCredentials(ctx)
		if err != nil {
			t.Fatalf("failed to get default credentials: %v", err)
		}

		// Create OAuth2 client
		oauth2Client := oauth2.NewClient(context.WithValue(ctx, oauth2.HTTPClient, hypertClient), creds.TokenSource)

		// Wrap the client to add quota project header
		return &http.Client{
			Transport: &quotaProjectTransport{
				base:      oauth2Client.Transport,
				projectID: projectID,
			},
			Timeout: oauth2Client.Timeout,
		}
	}

	return hypertClient
}

// GeminiTestConfig configures Gemini client creation for tests
type GeminiTestConfig struct {
	Project  string
	Location string
	SubDir   string // Subdirectory for hypert test data
}

// DefaultGeminiTestConfig returns a default configuration for Gemini testing
func DefaultGeminiTestConfig(subDir string) GeminiTestConfig {
	return GeminiTestConfig{
		Project:  os.Getenv("GOOGLE_PROJECT_ID"),
		Location: os.Getenv("GOOGLE_REGION"),
		SubDir:   subDir,
	}
}

// NewGeminiClient creates a new Gemini client for testing with hypert caching
func NewGeminiClient(t *testing.T, config GeminiTestConfig) *genai.Client {
	ctx := context.Background()

	// Create hypert client for caching
	hypertClient := NewHypertClient(t, HypertClientConfig{
		TestDataDir: "testdata",
		SubDir:      config.SubDir,
	})

	// Create Gemini client
	genaiClient, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend:    genai.BackendVertexAI,
		Project:    config.Project,
		Location:   config.Location,
		HTTPClient: hypertClient,
	})
	if err != nil {
		t.Fatalf("failed to create genai client: %v", err)
	}

	return genaiClient
}

// NewGeminiGenerator creates a new Gemini generator for testing
func NewGeminiGenerator(t *testing.T, config GeminiTestConfig, modelName string) *gemini.Generator {
	genaiClient := NewGeminiClient(t, config)
	return gemini.NewGenerator(genaiClient, modelName)
}

// NewGeminiEmbedder creates a new Gemini embedder for testing
func NewGeminiEmbedder(t *testing.T, config GeminiTestConfig, modelName string) *gemini.Embedder {
	genaiClient := NewGeminiClient(t, config)
	return gemini.NewEmbedder(genaiClient, modelName)
}
