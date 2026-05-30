package middleware

import (
	"context"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
)

// RequestHook is called before a request is sent to the provider.
type RequestHook func(ctx context.Context, req *llmrouter.Request)

// ResponseHook is called after a request completes (or fails).
// For streaming, it fires when EventDone is received, so latency covers the full stream.
// resp is nil when err is non-nil. Use llmrouter.CalculateCost to compute cost inside the hook.
type ResponseHook func(ctx context.Context, req *llmrouter.Request, resp *llmrouter.Response, err error, latency time.Duration)

// HooksOptions configures the hooks middleware.
type HooksOptions struct {
	OnRequest  RequestHook
	OnResponse ResponseHook
}

// NewHooks returns a MiddlewareFunc that fires OnRequest before and OnResponse after each call.
// Either hook may be nil. No observability platform is imported; callers wire slog, OTEL, etc.
func NewHooks(opts HooksOptions) llmrouter.MiddlewareFunc {
	return func(next llmrouter.Provider) llmrouter.Provider {
		return &hooksProvider{Provider: next, opts: opts}
	}
}

type hooksProvider struct {
	llmrouter.Provider
	opts HooksOptions
}

func (p *hooksProvider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	if p.opts.OnRequest != nil {
		p.opts.OnRequest(ctx, req)
	}
	start := time.Now()
	resp, err := p.Provider.Complete(ctx, req)
	if p.opts.OnResponse != nil {
		p.opts.OnResponse(context.WithoutCancel(ctx), req, resp, err, time.Since(start))
	}
	return resp, err
}

func (p *hooksProvider) Stream(ctx context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
	if p.opts.OnRequest != nil {
		p.opts.OnRequest(ctx, req)
	}
	start := time.Now()
	result, err := p.Provider.Stream(ctx, req)
	if err != nil {
		if p.opts.OnResponse != nil {
			p.opts.OnResponse(context.WithoutCancel(ctx), req, nil, err, time.Since(start))
		}
		return nil, err
	}
	if p.opts.OnResponse == nil {
		return result, nil
	}
	return wrapHookStream(ctx, result, req, p.opts.OnResponse, start), nil
}

// wrapHookStream wraps a StreamResult so that OnResponse fires when the stream ends.
func wrapHookStream(
	ctx context.Context,
	inner *llmrouter.StreamResult,
	req *llmrouter.Request,
	hook ResponseHook,
	start time.Time,
) *llmrouter.StreamResult {
	ch := make(chan llmrouter.Event)
	sr := llmrouter.NewStreamResult(ch)
	sr.OnClose(inner.Close)
	hookCtx := context.WithoutCancel(ctx)
	go func() {
		defer close(ch)
		for inner.Next() {
			event := inner.Event()
			if event.Type == llmrouter.EventDone {
				hook(hookCtx, req, event.Response, nil, time.Since(start))
				ch <- event
				return
			}
			ch <- event
		}
		if err := inner.Err(); err != nil {
			hook(hookCtx, req, nil, err, time.Since(start))
			ch <- llmrouter.Event{Type: llmrouter.EventError, Error: err}
		}
	}()
	return sr
}
