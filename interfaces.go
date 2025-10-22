package goeval

import (
	"github.com/datar-psa/goeval/api"
)

type LLMGenerator = api.LLMGenerator
type Embedder = api.Embedder
type ModerationProvider = api.ModerationProvider
type ModerationCategory = api.ModerationCategory
type ModerationResult = api.ModerationResult

var ModerationCategories = api.ModerationCategories
