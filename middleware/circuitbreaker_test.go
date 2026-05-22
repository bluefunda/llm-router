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
	"testing"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
	"github.com/bluefunda/llmrouter/middleware"
)

// failingProvider always returns the given error from Complete and Stream.
type failingProvider struct {
	err error
}

func (f *failingProvider) Name() string     { return "failing" }
func (f *failingProvider) Models() []string { return nil }

func (f *failingProvider) Complete(_ context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	return nil, f.err
}

func (f *failingProvider) Stream(_ context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
	return nil, f.err
}

// succeedingProvider always succeeds.
type succeedingProvider struct{}

func (s *succeedingProvider) Name() string     { return "succeeding" }
func (s *succeedingProvider) Models() []string { return nil }

func (s *succeedingProvider) Complete(_ context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	return &llmrouter.Response{ID: "ok"}, nil
}

func (s *succeedingProvider) Stream(_ context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
	ch := make(chan llmrouter.Event, 1)
	ch <- llmrouter.Event{Type: llmrouter.EventDone}
	close(ch)
	return llmrouter.NewStreamResult(ch), nil
}

func TestCircuitBreaker_ClosedOnSuccess(t *testing.T) {
	cb := middleware.NewCircuitBreaker(3, 30*time.Second)
	wrapped := cb.Wrap(&succeedingProvider{})

	for i := 0; i < 5; i++ {
		_, err := wrapped.Complete(t.Context(), minimalRequest())
		if err != nil {
			t.Fatalf("call %d: unexpected error: %v", i+1, err)
		}
	}

	if got := cb.State(); got != middleware.CBStateClosed {
		t.Errorf("expected circuit closed after successes, got %v", got)
	}
}

func TestCircuitBreaker_OpensAfterFailures(t *testing.T) {
	const maxFailures uint32 = 3
	providerErr := errors.New("provider failure")
	cb := middleware.NewCircuitBreaker(maxFailures, 30*time.Second)
	wrapped := cb.Wrap(&failingProvider{err: providerErr})

	// maxFailures+1 failures are needed to trip the breaker
	// (the counter must exceed maxFailures, i.e. be > maxFailures).
	for i := uint32(0); i <= maxFailures; i++ {
		_, _ = wrapped.Complete(t.Context(), minimalRequest())
	}

	if got := cb.State(); got != middleware.CBStateOpen {
		t.Errorf("expected circuit open after %d failures, got %v", maxFailures+1, got)
	}
}

func TestCircuitBreaker_OpenReturnsErrCircuitOpen(t *testing.T) {
	const maxFailures uint32 = 2
	providerErr := errors.New("provider failure")

	// countingProvider tracks calls so we can verify the breaker is short-circuiting.
	counter := &stubProvider{
		name: "counter",
		completeFn: func(n int) (*llmrouter.Response, error) {
			return nil, providerErr
		},
	}

	cb := middleware.NewCircuitBreaker(maxFailures, 30*time.Second)
	wrapped := cb.Wrap(counter)

	// Trip the circuit open.
	for i := uint32(0); i <= maxFailures; i++ {
		_, _ = wrapped.Complete(t.Context(), minimalRequest())
	}

	if got := cb.State(); got != middleware.CBStateOpen {
		t.Fatalf("circuit should be open, got %v", got)
	}

	callsBefore := counter.calls.Load()

	// Calls while open must return ErrCircuitOpen without reaching the provider.
	_, err := wrapped.Complete(t.Context(), minimalRequest())
	if !errors.Is(err, llmrouter.ErrCircuitOpen) {
		t.Errorf("expected ErrCircuitOpen, got %v", err)
	}
	if got := counter.calls.Load(); got != callsBefore {
		t.Errorf("provider should not be called when circuit is open: calls went from %d to %d", callsBefore, got)
	}
}

func TestCircuitBreaker_RecoveryAfterTimeout(t *testing.T) {
	const maxFailures uint32 = 2
	providerErr := errors.New("provider failure")

	// Use a very short open timeout so we don't have to wait long.
	const openTimeout = 20 * time.Millisecond
	cb := middleware.NewCircuitBreaker(maxFailures, openTimeout)

	// Use a stub so we can swap its behaviour between fail and succeed.
	stub := &stubProvider{name: "recoverable"}
	stub.completeFn = func(n int) (*llmrouter.Response, error) {
		return nil, providerErr
	}
	wrapped := cb.Wrap(stub)

	// Trip the circuit open: need maxFailures+1 consecutive failures.
	for i := uint32(0); i <= maxFailures; i++ {
		_, _ = wrapped.Complete(t.Context(), minimalRequest())
	}
	if got := cb.State(); got != middleware.CBStateOpen {
		t.Fatalf("circuit should be open, got %v", got)
	}

	// A request while open must fail immediately with ErrCircuitOpen.
	_, err := wrapped.Complete(t.Context(), minimalRequest())
	if !errors.Is(err, llmrouter.ErrCircuitOpen) {
		t.Fatalf("expected ErrCircuitOpen while open, got %v", err)
	}

	// Wait for the open timeout to elapse so the breaker moves to HalfOpen.
	time.Sleep(openTimeout + 10*time.Millisecond)

	// State() triggers the Open→HalfOpen transition.
	if got := cb.State(); got != middleware.CBStateHalfOpen {
		t.Errorf("expected half-open after timeout, got %v", got)
	}

	// Switch the stub to succeed so the probe request closes the circuit.
	stub.completeFn = func(n int) (*llmrouter.Response, error) {
		return &llmrouter.Response{ID: "recovered"}, nil
	}

	resp, err := wrapped.Complete(t.Context(), minimalRequest())
	if err != nil {
		t.Fatalf("probe request failed: %v", err)
	}
	if resp.ID != "recovered" {
		t.Errorf("expected response ID %q, got %q", "recovered", resp.ID)
	}
	if got := cb.State(); got != middleware.CBStateClosed {
		t.Errorf("expected circuit closed after successful probe, got %v", got)
	}
}
