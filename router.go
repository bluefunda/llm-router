package llmrouter

import (
	"context"
	"fmt"
	"sync"
)

// Router manages multiple LLM providers and routes requests
type Router struct {
	providers  map[string]Provider
	modelMap   map[string]string // model -> provider mapping
	fallbacks  []string          // ordered fallback providers
	middleware []Middleware
	mu         sync.RWMutex
}

// New creates a new Router with the given options
func New(opts ...Option) *Router {
	r := &Router{
		providers: make(map[string]Provider),
		modelMap:  make(map[string]string),
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Route sends a request to the appropriate provider and streams the response
func (r *Router) Route(ctx context.Context, req *Request) (<-chan Event, error) {
	provider, err := r.resolveProvider(req.Model)
	if err != nil {
		return nil, err
	}

	// Apply middleware chain
	handler := r.buildChain(provider)

	return handler.Stream(ctx, req)
}

// Complete performs a non-streaming completion
func (r *Router) Complete(ctx context.Context, req *Request) (*Response, error) {
	provider, err := r.resolveProvider(req.Model)
	if err != nil {
		return nil, err
	}

	handler := r.buildChain(provider)
	return handler.Complete(ctx, req)
}

// Stream is an alias for Route for clarity
func (r *Router) Stream(ctx context.Context, req *Request) (<-chan Event, error) {
	return r.Route(ctx, req)
}

// resolveProvider finds the right provider for a model
func (r *Router) resolveProvider(model string) (Provider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.providers) == 0 {
		return nil, ErrNoProviders
	}

	// Check explicit model mapping first
	if providerName, ok := r.modelMap[model]; ok {
		if p, ok := r.providers[providerName]; ok {
			return p, nil
		}
	}

	// Check if model name matches a provider name directly
	if p, ok := r.providers[model]; ok {
		return p, nil
	}

	// Try each provider to see if it supports this model
	for _, p := range r.providers {
		for _, m := range p.Models() {
			if m == model {
				return p, nil
			}
		}
	}

	return nil, fmt.Errorf("%w: %s", ErrUnknownModel, model)
}

// buildChain wraps the provider with middleware
func (r *Router) buildChain(provider Provider) Provider {
	result := provider
	// Apply middleware in reverse order so first middleware is outermost
	for i := len(r.middleware) - 1; i >= 0; i-- {
		result = r.middleware[i].Wrap(result)
	}
	return result
}

// RegisterProvider adds a provider to the router
func (r *Router) RegisterProvider(name string, p Provider) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.providers[name] = p
}

// MapModel maps a model name to a specific provider
func (r *Router) MapModel(model, provider string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.modelMap[model] = provider
}

// Providers returns list of registered provider names
func (r *Router) Providers() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.providers))
	for name := range r.providers {
		names = append(names, name)
	}
	return names
}

// GetProvider returns a provider by name
func (r *Router) GetProvider(name string) (Provider, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	p, ok := r.providers[name]
	return p, ok
}

// SetFallbacks sets the fallback provider order
func (r *Router) SetFallbacks(providers ...string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.fallbacks = providers
}

// AddMiddleware adds middleware to the router
func (r *Router) AddMiddleware(m Middleware) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.middleware = append(r.middleware, m)
}
