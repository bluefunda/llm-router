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

package middleware

import (
	"context"
	"errors"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
)

// CircuitBreakerMiddleware wraps a Provider with circuit breaker protection.
// The circuit breaker state machine lives in breaker.go.
type CircuitBreakerMiddleware struct {
	cb *circuitBreaker
}

// NewCircuitBreakerMiddleware creates a new circuit breaker middleware.
// It trips open after maxFailures consecutive failures and recovers after timeout.
func NewCircuitBreakerMiddleware(maxFailures uint32, timeout time.Duration) *CircuitBreakerMiddleware {
	return &CircuitBreakerMiddleware{cb: newCircuitBreaker(maxFailures, timeout)}
}

// State returns the current circuit breaker state.
func (m *CircuitBreakerMiddleware) State() CBState {
	return m.cb.State()
}

// Wrap wraps a provider with circuit breaker protection.
func (m *CircuitBreakerMiddleware) Wrap(next llmrouter.Provider) llmrouter.Provider {
	return &circuitBreakerProvider{Provider: next, cb: m.cb}
}

type circuitBreakerProvider struct {
	llmrouter.Provider
	cb *circuitBreaker
}

func (p *circuitBreakerProvider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	result, err := p.cb.Execute(func() (interface{}, error) {
		return p.Provider.Complete(ctx, req)
	})
	if err != nil {
		if errors.Is(err, errBreakerOpen) {
			return nil, llmrouter.ErrCircuitOpen
		}
		return nil, err
	}
	return result.(*llmrouter.Response), nil
}

func (p *circuitBreakerProvider) Stream(ctx context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
	result, err := p.cb.Execute(func() (interface{}, error) {
		return p.Provider.Stream(ctx, req)
	})
	if err != nil {
		if errors.Is(err, errBreakerOpen) {
			return nil, llmrouter.ErrCircuitOpen
		}
		return nil, err
	}
	return result.(*llmrouter.StreamResult), nil
}
