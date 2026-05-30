package llmrouter

import (
	"context"
)

// Provider is the core interface that all LLM providers must implement.
type Provider interface {
	// Name returns the provider identifier (e.g., "openai", "anthropic")
	Name() string

	// Models returns the list of supported model IDs
	Models() []string

	// Complete performs a non-streaming completion
	Complete(ctx context.Context, req *Request) (*Response, error)

	// Stream performs a streaming completion
	Stream(ctx context.Context, req *Request) (*StreamResult, error)
}

// MiddlewareFunc wraps a Provider with additional functionality.
// It is a plain function type; any func(Provider) Provider satisfies it directly.
type MiddlewareFunc func(Provider) Provider
