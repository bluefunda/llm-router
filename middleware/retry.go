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
	"fmt"
	"math"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
)

// RetryOption configures Retry middleware.
type RetryOption func(*retryConfig)

type retryConfig struct {
	maxDelay  time.Duration
	retryable func(error) bool
}

// WithMaxDelay sets the maximum delay between retries.
func WithMaxDelay(d time.Duration) RetryOption {
	return func(c *retryConfig) { c.maxDelay = d }
}

// WithRetryFunc sets a custom retry decision function.
func WithRetryFunc(f func(error) bool) RetryOption {
	return func(c *retryConfig) { c.retryable = f }
}

// Retry returns a MiddlewareFunc that retries failed requests with exponential backoff.
// Non-retryable errors (auth failures, invalid requests, context cancellation)
// short-circuit immediately without consuming retry attempts.
func Retry(maxAttempts int, baseDelay time.Duration, opts ...RetryOption) llmrouter.MiddlewareFunc {
	cfg := &retryConfig{
		maxDelay:  30 * time.Second,
		retryable: llmrouter.IsRetryable,
	}
	for _, o := range opts {
		o(cfg)
	}
	return func(next llmrouter.Provider) llmrouter.Provider {
		return &retryProvider{
			Provider:    next,
			maxAttempts: maxAttempts,
			baseDelay:   baseDelay,
			maxDelay:    cfg.maxDelay,
			retryable:   cfg.retryable,
		}
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

	return nil, fmt.Errorf("%w: %v", llmrouter.ErrMaxRetriesExceeded, lastErr)
}

func (p *retryProvider) Stream(ctx context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
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

		res, err := p.Provider.Stream(ctx, req)
		if err == nil {
			return res, nil
		}

		lastErr = err
		if !p.retryable(err) {
			return nil, err
		}
	}

	return nil, fmt.Errorf("%w: %v", llmrouter.ErrMaxRetriesExceeded, lastErr)
}

func (p *retryProvider) calculateBackoff(attempt int) time.Duration {
	delay := time.Duration(float64(p.baseDelay) * math.Pow(2, float64(attempt-1)))
	if delay > p.maxDelay {
		delay = p.maxDelay
	}
	return delay
}
