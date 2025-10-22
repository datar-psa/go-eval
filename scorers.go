package goeval

import (
	language "cloud.google.com/go/language/apiv1"
	"github.com/datar-psa/goeval/api"
	"github.com/datar-psa/goeval/embedding"
	"github.com/datar-psa/goeval/gemini"
	"github.com/datar-psa/goeval/heuristic"
	"github.com/datar-psa/goeval/llmjudge"
	"google.golang.org/genai"
)

type Score = api.Score
type ScoreInputs = api.ScoreInputs
type Scorer = api.Scorer

// LLMJudge wraps an LLM generator and exposes convenient constructors for LLM-as-a-judge scorers.
// It allows creating scorers like Factuality and Tonality without passing the LLM each time.
type LLMJudge struct {
	llm        api.LLMGenerator
	moderation api.ModerationProvider
}

// LLMJudgeOptions configures LLMJudge creation
type LLMJudgeOptions struct {
	llm        api.LLMGenerator
	moderation api.ModerationProvider
}

// WithLLMGenerator sets the LLM generator for the judge
func WithLLMGenerator(llm api.LLMGenerator) func(*LLMJudgeOptions) {
	return func(opts *LLMJudgeOptions) {
		opts.llm = llm
	}
}

// WithModerationProvider sets the moderation provider for the judge
func WithModerationProvider(provider api.ModerationProvider) func(*LLMJudgeOptions) {
	return func(opts *LLMJudgeOptions) {
		opts.moderation = provider
	}
}

// NewLLMJudge creates a new Judge wrapper using functional options.
func NewLLMJudge(opts ...func(*LLMJudgeOptions)) *LLMJudge {
	options := &LLMJudgeOptions{}
	for _, opt := range opts {
		opt(options)
	}
	return &LLMJudge{
		llm:        options.llm,
		moderation: options.moderation,
	}
}

// GeminiOptions configures Gemini LLMJudge creation
type GeminiOptions struct {
	genaiClient *genai.Client
	modelName   string
	langClient  *language.Client
}

// WithGenaiClient sets the Gemini client for the judge
func WithGenaiClient(client *genai.Client) func(*GeminiOptions) {
	return func(opts *GeminiOptions) {
		opts.genaiClient = client
	}
}

// WithModelName sets the model name for the judge
func WithModelName(modelName string) func(*GeminiOptions) {
	return func(opts *GeminiOptions) {
		opts.modelName = modelName
	}
}

// WithLanguageClient sets the Google Cloud Language client for moderation
func WithLanguageClient(langClient *language.Client) func(*GeminiOptions) {
	return func(opts *GeminiOptions) {
		opts.langClient = langClient
	}
}

// NewGeminiLLMJudge creates a Judge using Gemini client and model name.
// Example model: "publishers/google/models/gemini-2.5-flash".
func NewGeminiLLMJudge(opts ...func(*GeminiOptions)) *LLMJudge {
	options := &GeminiOptions{}
	for _, opt := range opts {
		opt(options)
	}

	var llmOptions []func(*LLMJudgeOptions)

	// Only add LLM generator if genaiClient is provided
	if options.genaiClient != nil && options.modelName != "" {
		llmOptions = append(llmOptions, WithLLMGenerator(gemini.NewGenerator(options.genaiClient, options.modelName)))
	}

	// Only add moderation provider if langClient is provided
	if options.langClient != nil {
		llmOptions = append(llmOptions, WithModerationProvider(gemini.NewGoogleLanguageProvider(options.langClient)))
	}

	return NewLLMJudge(llmOptions...)
}

type FactualityOptions = llmjudge.FactualityOptions

// Factuality returns a scorer that compares Output against Expected for factual consistency.
func (j *LLMJudge) Factuality(opts FactualityOptions) api.Scorer {
	return llmjudge.Factuality(j.llm, opts)
}

type TonalityOptions = llmjudge.TonalityOptions

// Tonality returns a scorer that evaluates professionalism, kindness, clarity and helpfulness.
func (j *LLMJudge) Tonality(opts TonalityOptions) api.Scorer {
	return llmjudge.Tonality(j.llm, opts)
}

type ModerationOptions = llmjudge.ModerationOptions

// Moderation returns a scorer that evaluates content safety using a moderation provider.
func (j *LLMJudge) Moderation(opts ModerationOptions) api.Scorer {
	return llmjudge.Moderation(j.moderation, opts)
}

// Embedding wraps an embedder and exposes convenient constructors for embedding-based scorers.
type Embedding struct{ embedder api.Embedder }

// EmbeddingOptions configures Embedding creation
type EmbeddingOptions struct {
	embedder api.Embedder
}

// WithEmbedder sets the embedder for the embedding scorer
func WithEmbedder(embedder api.Embedder) func(*EmbeddingOptions) {
	return func(opts *EmbeddingOptions) {
		opts.embedder = embedder
	}
}

// NewEmbedding creates a new Embedding wrapper using functional options.
func NewEmbedding(opts ...func(*EmbeddingOptions)) *Embedding {
	options := &EmbeddingOptions{}
	for _, opt := range opts {
		opt(options)
	}
	return &Embedding{embedder: options.embedder}
}

// NewGeminiEmbedding creates an Embedding using Gemini client and model name.
// Example model: "text-embedding-005".
func NewGeminiEmbedding(opts ...func(*GeminiOptions)) *Embedding {
	options := &GeminiOptions{}
	for _, opt := range opts {
		opt(options)
	}

	var embeddingOptions []func(*EmbeddingOptions)

	// Only add embedder if genaiClient and modelName are provided
	if options.genaiClient != nil && options.modelName != "" {
		embeddingOptions = append(embeddingOptions, WithEmbedder(gemini.NewEmbedder(options.genaiClient, options.modelName)))
	}

	return NewEmbedding(embeddingOptions...)
}

type EmbeddingSimilarityOptions = embedding.EmbeddingSimilarityOptions

// Similarity returns a scorer that measures semantic similarity using embeddings.
func (e *Embedding) Similarity(opts EmbeddingSimilarityOptions) api.Scorer {
	return embedding.EmbeddingSimilarity(e.embedder, opts)
}

// Heuristic exposes convenient constructors for heuristic scorers.
type Heuristic struct{}

// NewHeuristic creates a new Heuristic.
func NewHeuristic() *Heuristic {
	return &Heuristic{}
}

type ExactMatchOptions = heuristic.ExactMatchOptions

// ExactMatch returns a scorer that checks if the output exactly matches the expected value.
func (h *Heuristic) ExactMatch(opts ExactMatchOptions) api.Scorer {
	return heuristic.ExactMatch(opts)
}
