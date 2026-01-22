package llmrouter

import (
	"context"
)

// Provider is the core interface that all LLM providers must implement
type Provider interface {
	// Name returns the provider identifier (e.g., "openai", "anthropic")
	Name() string

	// Models returns the list of supported model IDs
	Models() []string

	// Complete performs a non-streaming completion
	Complete(ctx context.Context, req *Request) (*Response, error)

	// Stream performs a streaming completion, returning events via channel
	Stream(ctx context.Context, req *Request) (<-chan Event, error)

	// SupportsTools returns whether the provider supports function/tool calling
	SupportsTools() bool
}

// Middleware wraps a Provider with additional functionality
type Middleware interface {
	Wrap(next Provider) Provider
}
