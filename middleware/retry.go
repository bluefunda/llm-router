package middleware

import (
	"context"
	"fmt"
	"math"
	"time"

	llmrouter "github.com/bluefunda/llm-router"
)

// RetryMiddleware provides retry logic with exponential backoff
type RetryMiddleware struct {
	maxAttempts int
	baseDelay   time.Duration
	maxDelay    time.Duration
	retryable   func(error) bool
}

// NewRetryMiddleware creates a new retry middleware
func NewRetryMiddleware(maxAttempts int, baseDelay time.Duration) *RetryMiddleware {
	return &RetryMiddleware{
		maxAttempts: maxAttempts,
		baseDelay:   baseDelay,
		maxDelay:    30 * time.Second,
		retryable:   llmrouter.IsRetryable,
	}
}

// WithMaxDelay sets the maximum delay between retries
func (m *RetryMiddleware) WithMaxDelay(d time.Duration) *RetryMiddleware {
	m.maxDelay = d
	return m
}

// WithRetryFunc sets a custom retry decision function
func (m *RetryMiddleware) WithRetryFunc(f func(error) bool) *RetryMiddleware {
	m.retryable = f
	return m
}

// Wrap wraps a provider with retry logic
func (m *RetryMiddleware) Wrap(next llmrouter.Provider) llmrouter.Provider {
	return &retryProvider{
		Provider:    next,
		maxAttempts: m.maxAttempts,
		baseDelay:   m.baseDelay,
		maxDelay:    m.maxDelay,
		retryable:   m.retryable,
	}
}

type retryProvider struct {
	llmrouter.Provider
	maxAttempts int
	baseDelay   time.Duration
	maxDelay    time.Duration
	retryable   func(error) bool
}

func (p *retryProvider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	var lastErr error

	for attempt := 0; attempt < p.maxAttempts; attempt++ {
		if attempt > 0 {
			delay := p.calculateBackoff(attempt)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		resp, err := p.Provider.Complete(ctx, req)
		if err == nil {
			return resp, nil
		}

		lastErr = err
		if !p.retryable(err) {
			return nil, err
		}
	}

	return nil, fmt.Errorf("%w: %v", llmrouter.ErrMaxRetriesExceed, lastErr)
}

func (p *retryProvider) Stream(ctx context.Context, req *llmrouter.Request) (<-chan llmrouter.Event, error) {
	var lastErr error

	for attempt := 0; attempt < p.maxAttempts; attempt++ {
		if attempt > 0 {
			delay := p.calculateBackoff(attempt)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		ch, err := p.Provider.Stream(ctx, req)
		if err == nil {
			return ch, nil
		}

		lastErr = err
		if !p.retryable(err) {
			return nil, err
		}
	}

	return nil, fmt.Errorf("%w: %v", llmrouter.ErrMaxRetriesExceed, lastErr)
}

func (p *retryProvider) calculateBackoff(attempt int) time.Duration {
	delay := time.Duration(float64(p.baseDelay) * math.Pow(2, float64(attempt-1)))
	if delay > p.maxDelay {
		delay = p.maxDelay
	}
	return delay
}
