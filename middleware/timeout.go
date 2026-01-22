package middleware

import (
	"context"
	"time"

	llmrouter "github.com/bluefunda/llm-router"
)

// TimeoutMiddleware adds timeout to requests
type TimeoutMiddleware struct {
	timeout time.Duration
}

// NewTimeoutMiddleware creates a new timeout middleware
func NewTimeoutMiddleware(timeout time.Duration) *TimeoutMiddleware {
	return &TimeoutMiddleware{
		timeout: timeout,
	}
}

// Wrap wraps a provider with timeout
func (m *TimeoutMiddleware) Wrap(next llmrouter.Provider) llmrouter.Provider {
	return &timeoutProvider{
		Provider: next,
		timeout:  m.timeout,
	}
}

type timeoutProvider struct {
	llmrouter.Provider
	timeout time.Duration
}

func (p *timeoutProvider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()

	return p.Provider.Complete(ctx, req)
}

func (p *timeoutProvider) Stream(ctx context.Context, req *llmrouter.Request) (<-chan llmrouter.Event, error) {
	ctx, cancel := context.WithTimeout(ctx, p.timeout)

	ch, err := p.Provider.Stream(ctx, req)
	if err != nil {
		cancel()
		return nil, err
	}

	// Wrap the channel to handle context cancellation
	outCh := make(chan llmrouter.Event)
	go func() {
		defer close(outCh)
		defer cancel()

		for {
			select {
			case <-ctx.Done():
				outCh <- llmrouter.Event{
					Type:  llmrouter.EventError,
					Error: ctx.Err(),
				}
				return
			case event, ok := <-ch:
				if !ok {
					return
				}
				select {
				case outCh <- event:
				case <-ctx.Done():
					outCh <- llmrouter.Event{
						Type:  llmrouter.EventError,
						Error: ctx.Err(),
					}
					return
				}
			}
		}
	}()

	return outCh, nil
}
