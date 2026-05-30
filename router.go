package llmrouter

import (
	"context"
	"fmt"
	"io"
	"sync"
)

// Router manages multiple LLM providers and routes requests
type Router struct {
	providers     map[string]Provider
	providerOrder []string // insertion order for deterministic resolution
	modelMap      map[string]string
	fallbacks     []string
	middleware    []MiddlewareFunc
	mu            sync.RWMutex
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

// Stream sends a request to the appropriate provider and streams the response.
func (r *Router) Stream(ctx context.Context, req *Request) (*StreamResult, error) {
	provider, err := r.resolveProvider(req.Model)
	if err != nil {
		return nil, err
	}

	handler := r.buildChain(provider)
	res, err := handler.Stream(ctx, req)
	if err != nil {
		return r.tryStreamFallbacks(ctx, req, err)
	}
	return res, nil
}

// Complete performs a non-streaming completion
func (r *Router) Complete(ctx context.Context, req *Request) (*Response, error) {
	provider, err := r.resolveProvider(req.Model)
	if err != nil {
		return nil, err
	}

	handler := r.buildChain(provider)
	resp, err := handler.Complete(ctx, req)
	if err != nil {
		return r.tryFallbacks(ctx, req, err)
	}
	return resp, nil
}

// resolveProvider finds the right provider for a model.
// Resolution order: explicit model mapping → provider name match → ordered scan.
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

	// Scan in insertion order for deterministic resolution
	for _, name := range r.providerOrder {
		p := r.providers[name]
		for _, m := range p.Models() {
			if m == model {
				return p, nil
			}
		}
	}

	return nil, fmt.Errorf("%w: %s", ErrUnknownModel, model)
}

// tryFallbacks attempts each fallback provider in order after a primary failure.
func (r *Router) tryFallbacks(ctx context.Context, req *Request, primaryErr error) (*Response, error) {
	r.mu.RLock()
	fallbacks := make([]string, len(r.fallbacks))
	copy(fallbacks, r.fallbacks)
	r.mu.RUnlock()

	lastErr := primaryErr
	for _, name := range fallbacks {
		r.mu.RLock()
		p, ok := r.providers[name]
		r.mu.RUnlock()
		if !ok {
			continue
		}
		resp, err := r.buildChain(p).Complete(ctx, req)
		if err == nil {
			return resp, nil
		}
		lastErr = err
	}
	return nil, lastErr
}

// tryStreamFallbacks attempts each fallback provider for streaming after a primary failure.
func (r *Router) tryStreamFallbacks(ctx context.Context, req *Request, primaryErr error) (*StreamResult, error) {
	r.mu.RLock()
	fallbacks := make([]string, len(r.fallbacks))
	copy(fallbacks, r.fallbacks)
	r.mu.RUnlock()

	lastErr := primaryErr
	for _, name := range fallbacks {
		r.mu.RLock()
		p, ok := r.providers[name]
		r.mu.RUnlock()
		if !ok {
			continue
		}
		res, err := r.buildChain(p).Stream(ctx, req)
		if err == nil {
			return res, nil
		}
		lastErr = err
	}
	return nil, lastErr
}

// buildChain wraps the provider with middleware.
// It snapshots the middleware slice under the lock to avoid a data race
// with concurrent AddMiddleware calls.
func (r *Router) buildChain(provider Provider) Provider {
	r.mu.RLock()
	mw := make([]MiddlewareFunc, len(r.middleware))
	copy(mw, r.middleware)
	r.mu.RUnlock()

	result := provider
	for i := len(mw) - 1; i >= 0; i-- {
		result = mw[i](result)
	}
	return result
}

// RegisterProvider adds a provider to the router
func (r *Router) RegisterProvider(name string, p Provider) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.providers[name]; !exists {
		r.providerOrder = append(r.providerOrder, name)
	}
	r.providers[name] = p
}

// MapModel maps a model name to a specific provider
func (r *Router) MapModel(model, provider string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.modelMap[model] = provider
}

// Providers returns list of registered provider names in insertion order
func (r *Router) Providers() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, len(r.providerOrder))
	copy(names, r.providerOrder)
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
func (r *Router) AddMiddleware(m MiddlewareFunc) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.middleware = append(r.middleware, m)
}

// Close releases resources held by registered providers that implement io.Closer.
// Call this when the router is no longer needed (e.g. on application shutdown).
func (r *Router) Close() error {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var firstErr error
	for _, p := range r.providers {
		if c, ok := p.(io.Closer); ok {
			if err := c.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	return firstErr
}
