package middleware_test

import (
	"context"
	"errors"
	"testing"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
	"github.com/bluefunda/llmrouter/middleware"
)

// blockingProvider blocks Complete until its context is cancelled, then returns
// the context error. Stream returns a channel that blocks forever.
type blockingProvider struct{}

func (b *blockingProvider) Name() string     { return "blocking" }
func (b *blockingProvider) Models() []string { return nil }

func (b *blockingProvider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func (b *blockingProvider) Stream(ctx context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
	// Return a channel that never sends anything; the caller must observe context cancellation.
	ch := make(chan llmrouter.Event)
	go func() {
		<-ctx.Done()
		ch <- llmrouter.Event{Type: llmrouter.EventError, Error: ctx.Err()}
		close(ch)
	}()
	return llmrouter.NewStreamResult(ch), nil
}

// instantProvider completes immediately.
type instantProvider struct{}

func (p *instantProvider) Name() string     { return "instant" }
func (p *instantProvider) Models() []string { return nil }

func (p *instantProvider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	return &llmrouter.Response{ID: "instant"}, nil
}

func (p *instantProvider) Stream(ctx context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
	ch := make(chan llmrouter.Event, 1)
	ch <- llmrouter.Event{Type: llmrouter.EventDone}
	close(ch)
	return llmrouter.NewStreamResult(ch), nil
}

func TestTimeout_CompleteSucceeds(t *testing.T) {
	wrapped := middleware.Timeout(5 * time.Second)(&instantProvider{})
	resp, err := wrapped.Complete(t.Context(), minimalRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ID != "instant" {
		t.Errorf("expected response ID %q, got %q", "instant", resp.ID)
	}
}

func TestTimeout_CompleteExceedsDeadline(t *testing.T) {
	// 10ms timeout — blockingProvider will never finish within that time.
	wrapped := middleware.Timeout(10 * time.Millisecond)(&blockingProvider{})
	_, err := wrapped.Complete(t.Context(), minimalRequest())
	if err == nil {
		t.Fatal("expected a timeout error, got nil")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("expected context.DeadlineExceeded, got %v", err)
	}
}

func TestTimeout_StreamSucceeds(t *testing.T) {
	wrapped := middleware.Timeout(5 * time.Second)(&instantProvider{})
	res, err := wrapped.Stream(t.Context(), minimalRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer res.Close() //nolint:errcheck

	// Drain the stream — should receive EventDone and no error.
	for res.Next() {
	}
	if err := res.Err(); err != nil {
		t.Errorf("unexpected stream error: %v", err)
	}
}

func TestTimeout_StreamExceedsDeadline(t *testing.T) {
	// 10ms timeout — blockingProvider's stream blocks until cancelled.
	wrapped := middleware.Timeout(10 * time.Millisecond)(&blockingProvider{})
	res, err := wrapped.Stream(t.Context(), minimalRequest())
	if err != nil {
		t.Fatalf("unexpected error starting stream: %v", err)
	}
	defer res.Close() //nolint:errcheck

	// Drain the stream; the blockingProvider pushes an EventError when cancelled.
	for res.Next() {
	}
	if err := res.Err(); err == nil {
		t.Error("expected a deadline error on stream, got nil")
	} else if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("expected context.DeadlineExceeded, got %v", err)
	}
}
