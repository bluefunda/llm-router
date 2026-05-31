package middleware

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
)

// stubProvider is a minimal Provider for testing.
type stubProvider struct {
	name     string
	completeResp *llmrouter.Response
	completeErr  error
	streamEvents []llmrouter.Event
	streamErr    error
}

func (s *stubProvider) Name() string   { return s.name }
func (s *stubProvider) Models() []string { return nil }

func (s *stubProvider) Complete(_ context.Context, _ *llmrouter.Request) (*llmrouter.Response, error) {
	return s.completeResp, s.completeErr
}

func (s *stubProvider) Stream(_ context.Context, _ *llmrouter.Request) (*llmrouter.StreamResult, error) {
	if s.streamErr != nil {
		return nil, s.streamErr
	}
	ch := make(chan llmrouter.Event, len(s.streamEvents))
	for _, e := range s.streamEvents {
		ch <- e
	}
	close(ch)
	return llmrouter.NewStreamResult(ch), nil
}

func TestHooks_OnRequest_Complete(t *testing.T) {
	var called int32
	stub := &stubProvider{
		name:         "stub",
		completeResp: &llmrouter.Response{Model: "m"},
	}
	mw := NewHooks(HooksOptions{
		OnRequest: func(_ context.Context, req *llmrouter.Request) {
			atomic.AddInt32(&called, 1)
		},
	})
	p := mw(stub)
	req := &llmrouter.Request{Model: "m"}
	_, _ = p.Complete(t.Context(), req)
	if atomic.LoadInt32(&called) != 1 {
		t.Errorf("OnRequest called %d times, want 1", called)
	}
}

func TestHooks_OnResponse_Complete_Success(t *testing.T) {
	wantResp := &llmrouter.Response{Model: "gpt-4o", Usage: &llmrouter.Usage{TotalTokens: 10}}
	stub := &stubProvider{name: "stub", completeResp: wantResp}

	var gotResp *llmrouter.Response
	var gotErr error
	var gotLatency time.Duration

	mw := NewHooks(HooksOptions{
		OnResponse: func(_ context.Context, _ *llmrouter.Request, resp *llmrouter.Response, err error, latency time.Duration) {
			gotResp = resp
			gotErr = err
			gotLatency = latency
		},
	})
	p := mw(stub)
	_, _ = p.Complete(t.Context(), &llmrouter.Request{})

	if gotResp != wantResp {
		t.Error("OnResponse received wrong response")
	}
	if gotErr != nil {
		t.Errorf("unexpected error in hook: %v", gotErr)
	}
	if gotLatency <= 0 {
		t.Error("expected positive latency")
	}
}

func TestHooks_OnResponse_Complete_Error(t *testing.T) {
	wantErr := errors.New("provider error")
	stub := &stubProvider{name: "stub", completeErr: wantErr}

	var gotResp *llmrouter.Response
	var gotErr error

	mw := NewHooks(HooksOptions{
		OnResponse: func(_ context.Context, _ *llmrouter.Request, resp *llmrouter.Response, err error, _ time.Duration) {
			gotResp = resp
			gotErr = err
		},
	})
	p := mw(stub)
	_, _ = p.Complete(t.Context(), &llmrouter.Request{})

	if gotResp != nil {
		t.Error("expected nil response on error")
	}
	if !errors.Is(gotErr, wantErr) {
		t.Errorf("expected %v, got %v", wantErr, gotErr)
	}
}

func TestHooks_OnResponse_Stream_Success(t *testing.T) {
	doneResp := &llmrouter.Response{Model: "claude-sonnet-4-20250514", Usage: &llmrouter.Usage{TotalTokens: 5}}
	stub := &stubProvider{
		name: "stub",
		streamEvents: []llmrouter.Event{
			{Type: llmrouter.EventContentDelta, Content: "hello"},
			{Type: llmrouter.EventDone, Response: doneResp},
		},
	}

	var gotResp *llmrouter.Response
	var gotErr error
	var gotLatency time.Duration

	mw := NewHooks(HooksOptions{
		OnResponse: func(_ context.Context, _ *llmrouter.Request, resp *llmrouter.Response, err error, latency time.Duration) {
			gotResp = resp
			gotErr = err
			gotLatency = latency
		},
	})
	p := mw(stub)
	sr, err := p.Stream(t.Context(), &llmrouter.Request{})
	if err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
	defer func() { _ = sr.Close() }()

	for sr.Next() {
	}
	if sr.Err() != nil {
		t.Fatalf("stream error: %v", sr.Err())
	}

	if gotResp != doneResp {
		t.Error("OnResponse received wrong response for stream")
	}
	if gotErr != nil {
		t.Errorf("unexpected error in stream hook: %v", gotErr)
	}
	if gotLatency <= 0 {
		t.Error("expected positive latency for stream")
	}
}

func TestHooks_OnResponse_Stream_Error(t *testing.T) {
	wantErr := errors.New("stream error")
	stub := &stubProvider{name: "stub", streamErr: wantErr}

	var gotErr error
	mw := NewHooks(HooksOptions{
		OnResponse: func(_ context.Context, _ *llmrouter.Request, _ *llmrouter.Response, err error, _ time.Duration) {
			gotErr = err
		},
	})
	p := mw(stub)
	_, err := p.Stream(t.Context(), &llmrouter.Request{})
	if !errors.Is(err, wantErr) {
		t.Errorf("expected stream error %v, got %v", wantErr, err)
	}
	if !errors.Is(gotErr, wantErr) {
		t.Errorf("OnResponse received wrong error: %v", gotErr)
	}
}

func TestHooks_NilHooks_NoPanic(t *testing.T) {
	stub := &stubProvider{
		name:         "stub",
		completeResp: &llmrouter.Response{},
		streamEvents: []llmrouter.Event{{Type: llmrouter.EventDone, Response: &llmrouter.Response{}}},
	}
	mw := NewHooks(HooksOptions{}) // both hooks nil
	p := mw(stub)

	if _, err := p.Complete(t.Context(), &llmrouter.Request{}); err != nil {
		t.Errorf("unexpected complete error: %v", err)
	}
	sr, err := p.Stream(t.Context(), &llmrouter.Request{})
	if err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
	for sr.Next() {
	}
}

func TestHooks_DelegatesNameAndModels(t *testing.T) {
	stub := &stubProvider{name: "my-provider"}
	p := NewHooks(HooksOptions{})(stub)
	if p.Name() != "my-provider" {
		t.Errorf("expected my-provider, got %s", p.Name())
	}
}
