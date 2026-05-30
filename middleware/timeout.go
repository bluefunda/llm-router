package middleware

import (
	"context"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
)

// Timeout returns a MiddlewareFunc that enforces a per-request context deadline.
// On timeout, Stream returns an error via StreamResult.Err() rather than blocking.
func Timeout(d time.Duration) llmrouter.MiddlewareFunc {
	return func(next llmrouter.Provider) llmrouter.Provider {
		return &timeoutProvider{Provider: next, timeout: d}
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

func (p *timeoutProvider) Stream(ctx context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
	ctx, cancel := context.WithTimeout(ctx, p.timeout)

	res, err := p.Provider.Stream(ctx, req)
	if err != nil {
		cancel()
		return nil, err
	}
	res.OnClose(func() error { cancel(); return nil })
	return res, nil
}
