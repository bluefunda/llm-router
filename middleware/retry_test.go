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

package middleware_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
	"github.com/bluefunda/llmrouter/middleware"
)

// stubProvider is a test double for llmrouter.Provider.
// completeFn and streamFn are called on each invocation and may be swapped
// between calls via the calls counter.
type stubProvider struct {
	name       string
	calls      atomic.Int32
	completeFn func(n int) (*llmrouter.Response, error)
	streamFn   func(n int) (*llmrouter.StreamResult, error)
}

func (s *stubProvider) Name() string   { return s.name }
func (s *stubProvider) Models() []string { return nil }

func (s *stubProvider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	n := int(s.calls.Add(1))
	if s.completeFn != nil {
		return s.completeFn(n)
	}
	return &llmrouter.Response{}, nil
}

func (s *stubProvider) Stream(ctx context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
	n := int(s.calls.Add(1))
	if s.streamFn != nil {
		return s.streamFn(n)
	}
	ch := make(chan llmrouter.Event, 1)
	ch <- llmrouter.Event{Type: llmrouter.EventDone}
	close(ch)
	return llmrouter.NewStreamResult(ch), nil
}

// retryableErr returns an *APIError with status 429 that IsRetryable() reports true for.
func retryableErr() error {
	return &llmrouter.APIError{StatusCode: 429, Err: llmrouter.ErrRateLimited}
}

// nonRetryableErr returns an *APIError with status 400 that IsRetryable() reports false for.
func nonRetryableErr() error {
	return &llmrouter.APIError{StatusCode: 400, Err: llmrouter.ErrInvalidRequest}
}

func minimalRequest() *llmrouter.Request {
	return &llmrouter.Request{
		Model:    "test-model",
		Messages: []llmrouter.Message{{Role: llmrouter.RoleUser, Content: "hi"}},
	}
}

func TestRetry_SucceedsOnFirstAttempt(t *testing.T) {
	stub := &stubProvider{
		name: "stub",
		completeFn: func(n int) (*llmrouter.Response, error) {
			return &llmrouter.Response{ID: "ok"}, nil
		},
	}

	wrapped := middleware.Retry(3, 0)(stub)
	resp, err := wrapped.Complete(t.Context(), minimalRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ID != "ok" {
		t.Errorf("expected response ID %q, got %q", "ok", resp.ID)
	}
	if got := stub.calls.Load(); got != 1 {
		t.Errorf("expected 1 call, got %d", got)
	}
}

func TestRetry_RetriesOnRetryableError(t *testing.T) {
	stub := &stubProvider{
		name: "stub",
		completeFn: func(n int) (*llmrouter.Response, error) {
			if n < 3 {
				return nil, retryableErr()
			}
			return &llmrouter.Response{ID: "ok"}, nil
		},
	}

	wrapped := middleware.Retry(3, time.Nanosecond)(stub)
	resp, err := wrapped.Complete(t.Context(), minimalRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ID != "ok" {
		t.Errorf("expected response ID %q, got %q", "ok", resp.ID)
	}
	if got := stub.calls.Load(); got != 3 {
		t.Errorf("expected exactly 3 calls, got %d", got)
	}
}

func TestRetry_NoRetryOnNonRetryable(t *testing.T) {
	stub := &stubProvider{
		name: "stub",
		completeFn: func(n int) (*llmrouter.Response, error) {
			return nil, nonRetryableErr()
		},
	}

	wrapped := middleware.Retry(3, time.Nanosecond)(stub)
	_, err := wrapped.Complete(t.Context(), minimalRequest())
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	if got := stub.calls.Load(); got != 1 {
		t.Errorf("expected exactly 1 call (no retry), got %d", got)
	}
	// Confirm it is NOT wrapped in ErrMaxRetriesExceeded
	if errors.Is(err, llmrouter.ErrMaxRetriesExceeded) {
		t.Errorf("non-retryable error should not be wrapped in ErrMaxRetriesExceeded")
	}
}

func TestRetry_ExhaustsAllAttempts(t *testing.T) {
	const maxAttempts = 4
	stub := &stubProvider{
		name: "stub",
		completeFn: func(n int) (*llmrouter.Response, error) {
			return nil, &llmrouter.APIError{StatusCode: 503, Err: llmrouter.ErrProviderError}
		},
	}

	wrapped := middleware.Retry(maxAttempts, time.Nanosecond)(stub)
	_, err := wrapped.Complete(t.Context(), minimalRequest())
	if err == nil {
		t.Fatal("expected an error after exhausting retries")
	}
	if !errors.Is(err, llmrouter.ErrMaxRetriesExceeded) {
		t.Errorf("expected ErrMaxRetriesExceeded, got %v", err)
	}
	if got := stub.calls.Load(); got != maxAttempts {
		t.Errorf("expected exactly %d calls, got %d", maxAttempts, got)
	}
}

func TestRetry_WithMaxDelay(t *testing.T) {
	// WithMaxDelay(0) means backoff is capped at 0 — no real sleeping.
	stub := &stubProvider{
		name: "stub",
		completeFn: func(n int) (*llmrouter.Response, error) {
			if n < 3 {
				return nil, retryableErr()
			}
			return &llmrouter.Response{ID: "done"}, nil
		},
	}

	wrapped := middleware.Retry(3, time.Nanosecond, middleware.WithMaxDelay(0))(stub)
	resp, err := wrapped.Complete(t.Context(), minimalRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ID != "done" {
		t.Errorf("expected response ID %q, got %q", "done", resp.ID)
	}
	if got := stub.calls.Load(); got != 3 {
		t.Errorf("expected exactly 3 calls, got %d", got)
	}
}

// customErr is a sentinel used in TestRetry_CustomRetryFunc.
var customErr = errors.New("custom retriable error")

func TestRetry_CustomRetryFunc(t *testing.T) {
	stub := &stubProvider{
		name: "stub",
		completeFn: func(n int) (*llmrouter.Response, error) {
			if n < 3 {
				return nil, customErr
			}
			return &llmrouter.Response{ID: "custom-ok"}, nil
		},
	}

	// Custom retry function: only retries on customErr.
	retryOnCustom := func(err error) bool {
		return errors.Is(err, customErr)
	}

	wrapped := middleware.Retry(3, time.Nanosecond, middleware.WithRetryFunc(retryOnCustom))(stub)
	resp, err := wrapped.Complete(t.Context(), minimalRequest())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ID != "custom-ok" {
		t.Errorf("expected response ID %q, got %q", "custom-ok", resp.ID)
	}
	if got := stub.calls.Load(); got != 3 {
		t.Errorf("expected exactly 3 calls, got %d", got)
	}

	// Verify non-custom errors are NOT retried by this function.
	stub2 := &stubProvider{
		name: "stub2",
		completeFn: func(n int) (*llmrouter.Response, error) {
			return nil, retryableErr() // a 429 — retryable by default but NOT by our custom func
		},
	}
	wrapped2 := middleware.Retry(3, time.Nanosecond, middleware.WithRetryFunc(retryOnCustom))(stub2)
	_, err2 := wrapped2.Complete(t.Context(), minimalRequest())
	if err2 == nil {
		t.Fatal("expected error when custom retry func rejects the error")
	}
	if got := stub2.calls.Load(); got != 1 {
		t.Errorf("expected 1 call (no retry), got %d", got)
	}
}
