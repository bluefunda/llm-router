package middleware

import (
	"context"
	"time"

	llmrouter "github.com/bluefunda/llm-router"
	"github.com/sony/gobreaker"
)

// CircuitBreakerMiddleware provides circuit breaker protection
type CircuitBreakerMiddleware struct {
	cb *gobreaker.CircuitBreaker
}

// NewCircuitBreakerMiddleware creates a new circuit breaker middleware
func NewCircuitBreakerMiddleware(name string, maxFailures uint32, timeout time.Duration) *CircuitBreakerMiddleware {
	cb := gobreaker.NewCircuitBreaker(gobreaker.Settings{
		Name:        name,
		MaxRequests: maxFailures,
		Interval:    60 * time.Second,
		Timeout:     timeout,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures > maxFailures
		},
		OnStateChange: func(name string, from gobreaker.State, to gobreaker.State) {
			// Could add logging here
		},
	})

	return &CircuitBreakerMiddleware{cb: cb}
}

// Wrap wraps a provider with circuit breaker protection
func (m *CircuitBreakerMiddleware) Wrap(next llmrouter.Provider) llmrouter.Provider {
	return &circuitBreakerProvider{
		Provider: next,
		cb:       m.cb,
	}
}

// State returns the current circuit breaker state
func (m *CircuitBreakerMiddleware) State() gobreaker.State {
	return m.cb.State()
}

type circuitBreakerProvider struct {
	llmrouter.Provider
	cb *gobreaker.CircuitBreaker
}

func (p *circuitBreakerProvider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	result, err := p.cb.Execute(func() (interface{}, error) {
		return p.Provider.Complete(ctx, req)
	})

	if err != nil {
		if err == gobreaker.ErrOpenState || err == gobreaker.ErrTooManyRequests {
			return nil, llmrouter.ErrCircuitOpen
		}
		return nil, err
	}

	return result.(*llmrouter.Response), nil
}

func (p *circuitBreakerProvider) Stream(ctx context.Context, req *llmrouter.Request) (<-chan llmrouter.Event, error) {
	result, err := p.cb.Execute(func() (interface{}, error) {
		return p.Provider.Stream(ctx, req)
	})

	if err != nil {
		if err == gobreaker.ErrOpenState || err == gobreaker.ErrTooManyRequests {
			return nil, llmrouter.ErrCircuitOpen
		}
		return nil, err
	}

	return result.(<-chan llmrouter.Event), nil
}
