// Copyright 2025 bluefunda
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

	handler := r.buildChain(provider)
	ch, err := handler.Stream(ctx, req)
	if err != nil {
		return r.tryStreamFallbacks(ctx, req, err)
	}
	return ch, nil
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
func (r *Router) tryStreamFallbacks(ctx context.Context, req *Request, primaryErr error) (<-chan Event, error) {
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
		ch, err := r.buildChain(p).Stream(ctx, req)
		if err == nil {
			return ch, nil
		}
		lastErr = err
	}
	return nil, lastErr
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
