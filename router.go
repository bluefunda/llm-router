package llmrouter

import (
	"context"
	"fmt"
	"io"
	"slices"
	"sync"
)

// Router manages multiple LLM providers and routes requests
type Router struct {
	providers     map[string]Provider
	providerOrder []string // insertion order for deterministic resolution
	modelMap      map[string]string
	fallbacks     []string
	middleware    []MiddlewareFunc
	priceTable    map[string]ModelPrice // nil means use DefaultPrices
	policy        RoutingPolicy         // nil means use built-in static resolution
	modelConfigs  map[string]ModelConfig
	mu            sync.RWMutex
}

// New creates a new Router with the given options
func New(opts ...Option) *Router {
	r := &Router{
		providers:    make(map[string]Provider),
		modelMap:     make(map[string]string),
		modelConfigs: make(map[string]ModelConfig),
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Stream sends a request to the appropriate provider and streams the response.
func (r *Router) Stream(ctx context.Context, req *Request) (*StreamResult, error) {
	provider, effectiveReq, err := r.resolveProvider(ctx, req)
	if err != nil {
		return nil, err
	}

	handler := r.buildChain(provider)
	res, err := handler.Stream(ctx, effectiveReq)
	if err != nil {
		return r.tryStreamFallbacks(ctx, effectiveReq, err)
	}
	return wrapStreamCost(res, r.prices()), nil
}

// Complete performs a non-streaming completion
func (r *Router) Complete(ctx context.Context, req *Request) (*Response, error) {
	provider, effectiveReq, err := r.resolveProvider(ctx, req)
	if err != nil {
		return nil, err
	}

	handler := r.buildChain(provider)
	resp, err := handler.Complete(ctx, effectiveReq)
	if err != nil {
		return r.tryFallbacks(ctx, effectiveReq, err)
	}
	if resp != nil && resp.Usage != nil {
		resp.Usage.Cost = CalculateCost(resp.Model, resp.Usage, r.prices())
	}
	return resp, nil
}

// prices returns the effective price table under the read lock.
func (r *Router) prices() map[string]ModelPrice {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.priceTable != nil {
		return r.priceTable
	}
	return DefaultPrices
}

// wrapStreamCost wraps a StreamResult so that EventDone.Response.Usage.Cost is populated.
func wrapStreamCost(inner *StreamResult, prices map[string]ModelPrice) *StreamResult {
	ch := make(chan Event)
	sr := NewStreamResult(ch)
	sr.OnClose(inner.Close)
	go func() {
		defer close(ch)
		for inner.Next() {
			event := inner.Event()
			if event.Type == EventDone && event.Response != nil && event.Response.Usage != nil {
				event.Response.Usage.Cost = CalculateCost(event.Response.Model, event.Response.Usage, prices)
			}
			ch <- event
			if event.Type == EventDone {
				return
			}
		}
		if err := inner.Err(); err != nil {
			ch <- Event{Type: EventError, Error: err}
		}
	}()
	return sr
}

// resolveProvider finds the right provider for a request. If a RoutingPolicy
// is configured, it is consulted over the full candidate set (every model
// across every registered provider) and may select a different model than
// the one requested; the returned Request reflects that selection. Otherwise
// resolution falls back to the router's built-in static resolution, and the
// request is returned unchanged.
func (r *Router) resolveProvider(ctx context.Context, req *Request) (Provider, *Request, error) {
	r.mu.RLock()
	policy := r.policy
	r.mu.RUnlock()

	if policy == nil {
		p, err := r.resolveProviderStatic(req.Model)
		return p, req, err
	}
	return r.resolveProviderWithPolicy(ctx, policy, req)
}

// resolveProviderStatic finds the right provider for a model.
// Resolution order: explicit model mapping → provider name match → ordered scan.
func (r *Router) resolveProviderStatic(model string) (Provider, error) {
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
		if slices.Contains(p.Models(), model) {
			return p, nil
		}
	}

	return nil, fmt.Errorf("%w: %s", ErrUnknownModel, model)
}

// resolveProviderWithPolicy builds the full candidate list from registered
// providers (merging in any tier/cost metadata from [Router.SetModelConfig]),
// asks the policy to pick one, and returns the provider it names along with
// a copy of req whose Model field reflects the policy's selection.
func (r *Router) resolveProviderWithPolicy(ctx context.Context, policy RoutingPolicy, req *Request) (Provider, *Request, error) {
	r.mu.RLock()
	if len(r.providers) == 0 {
		r.mu.RUnlock()
		return nil, nil, ErrNoProviders
	}
	candidates := make([]ModelConfig, 0, len(r.providers))
	for _, name := range r.providerOrder {
		p := r.providers[name]
		for _, m := range p.Models() {
			cfg := r.modelConfigs[m]
			cfg.Provider = name
			cfg.Model = m
			candidates = append(candidates, cfg)
		}
	}
	r.mu.RUnlock()

	selected, err := policy.SelectModel(ctx, RoutingQuery{
		Model:    req.Model,
		Messages: req.Messages,
		Metadata: req.Metadata,
	}, candidates)
	if err != nil {
		return nil, nil, err
	}

	r.mu.RLock()
	p, ok := r.providers[selected.Provider]
	r.mu.RUnlock()
	if !ok {
		return nil, nil, fmt.Errorf("%w: %s", ErrUnknownProvider, selected.Provider)
	}

	effectiveReq := req
	if selected.Model != req.Model {
		clone := *req
		clone.Model = selected.Model
		effectiveReq = &clone
	}
	return p, effectiveReq, nil
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

// SetRoutingPolicy sets the RoutingPolicy used to select among candidate
// models. Pass nil (the default) to restore the router's built-in static
// resolution.
func (r *Router) SetRoutingPolicy(p RoutingPolicy) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.policy = p
}

// SetModelConfig registers tier/cost metadata for a model, consulted by
// RoutingPolicy implementations via the candidate list passed to
// SelectModel. Has no effect on the router's built-in static resolution.
func (r *Router) SetModelConfig(cfg ModelConfig) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.modelConfigs[cfg.Model] = cfg
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
