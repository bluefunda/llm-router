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

package llmrouter

import (
	"context"
	"errors"
	"testing"
)

type mockProvider struct {
	name   string
	models []string
	err    error
	resp   *Response
}

func (m *mockProvider) Name() string     { return m.name }
func (m *mockProvider) Models() []string { return m.models }

func (m *mockProvider) Complete(_ context.Context, _ *Request) (*Response, error) {
	return m.resp, m.err
}

func (m *mockProvider) Stream(_ context.Context, _ *Request) (*StreamResult, error) {
	if m.err != nil {
		return nil, m.err
	}
	ch := make(chan Event, 1)
	ch <- Event{Type: EventDone}
	close(ch)
	return NewStreamResult(ch), nil
}

func TestFallbackComplete(t *testing.T) {
	primary := &mockProvider{name: "primary", models: []string{"gpt-4o"}, err: ErrProviderError}
	fallback := &mockProvider{name: "fallback", models: []string{"gpt-4o-mini"}, resp: &Response{Model: "gpt-4o-mini"}}

	r := New(
		WithProvider("primary", primary),
		WithProvider("fallback", fallback),
		WithModelMapping("gpt-4o", "primary"),
		WithFallback("fallback"),
	)

	resp, err := r.Complete(t.Context(), &Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("expected fallback success, got: %v", err)
	}
	if resp.Model != "gpt-4o-mini" {
		t.Errorf("expected fallback model gpt-4o-mini, got %s", resp.Model)
	}
}

func TestFallbackStream(t *testing.T) {
	primary := &mockProvider{name: "primary", models: []string{"gpt-4o"}, err: ErrProviderError}
	fallback := &mockProvider{name: "fallback", models: []string{"gpt-4o-mini"}, resp: &Response{}}

	r := New(
		WithProvider("primary", primary),
		WithProvider("fallback", fallback),
		WithModelMapping("gpt-4o", "primary"),
		WithFallback("fallback"),
	)

	stream, err := r.Stream(t.Context(), &Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("expected fallback success, got: %v", err)
	}
	defer stream.Close() //nolint:errcheck
	for stream.Next() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
}

func TestFallbackAllFail(t *testing.T) {
	primary := &mockProvider{name: "primary", models: []string{"gpt-4o"}, err: ErrProviderError}
	fallback := &mockProvider{name: "fallback", models: []string{"gpt-4o-mini"}, err: ErrRateLimited}

	r := New(
		WithProvider("primary", primary),
		WithProvider("fallback", fallback),
		WithModelMapping("gpt-4o", "primary"),
		WithFallback("fallback"),
	)

	_, err := r.Complete(t.Context(), &Request{Model: "gpt-4o"})
	if err == nil {
		t.Fatal("expected error when all fallbacks fail")
	}
}

func TestNoFallbackReturnsError(t *testing.T) {
	primary := &mockProvider{name: "primary", models: []string{"gpt-4o"}, err: ErrProviderError}

	r := New(
		WithProvider("primary", primary),
		WithModelMapping("gpt-4o", "primary"),
	)

	_, err := r.Complete(t.Context(), &Request{Model: "gpt-4o"})
	if err == nil {
		t.Fatal("expected error when no fallback configured")
	}
}

func TestFallbackUnknownProviderSkipped(t *testing.T) {
	primary := &mockProvider{name: "primary", models: []string{"gpt-4o"}, err: ErrProviderError}
	good := &mockProvider{name: "good", models: []string{"gpt-4o-mini"}, resp: &Response{Model: "gpt-4o-mini"}}

	r := New(
		WithProvider("primary", primary),
		WithProvider("good", good),
		WithModelMapping("gpt-4o", "primary"),
		WithFallback("nonexistent", "good"), // nonexistent is skipped, good succeeds
	)

	resp, err := r.Complete(t.Context(), &Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("expected success after skipping unknown fallback, got: %v", err)
	}
	if resp.Model != "gpt-4o-mini" {
		t.Errorf("expected gpt-4o-mini, got %s", resp.Model)
	}
}

// --- Model resolution tests ---

func TestModelResolution_ExplicitMapping(t *testing.T) {
	openai := &mockProvider{name: "openai", models: []string{"gpt-4o"}, resp: &Response{Model: "gpt-4o"}}

	r := New(
		WithProvider("openai", openai),
		WithModelMapping("gpt-4o", "openai"),
	)

	resp, err := r.Complete(t.Context(), &Request{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("expected success with explicit mapping, got: %v", err)
	}
	if resp.Model != "gpt-4o" {
		t.Errorf("expected gpt-4o, got %s", resp.Model)
	}
}

func TestModelResolution_ProviderNameMatch(t *testing.T) {
	// When Model == provider name, the provider name-match step should resolve it.
	openai := &mockProvider{name: "openai", models: []string{}, resp: &Response{Model: "openai"}}

	r := New(
		WithProvider("openai", openai),
	)

	resp, err := r.Complete(t.Context(), &Request{Model: "openai"})
	if err != nil {
		t.Fatalf("expected success via provider name match, got: %v", err)
	}
	if resp.Model != "openai" {
		t.Errorf("expected openai, got %s", resp.Model)
	}
}

func TestModelResolution_ModelListScan(t *testing.T) {
	// No explicit mapping, model name != provider name, but provider lists the model.
	openai := &mockProvider{name: "openai", models: []string{"gpt-4o-mini"}, resp: &Response{Model: "gpt-4o-mini"}}

	r := New(
		WithProvider("openai", openai),
	)

	resp, err := r.Complete(t.Context(), &Request{Model: "gpt-4o-mini"})
	if err != nil {
		t.Fatalf("expected success via model list scan, got: %v", err)
	}
	if resp.Model != "gpt-4o-mini" {
		t.Errorf("expected gpt-4o-mini, got %s", resp.Model)
	}
}

func TestModelResolution_UnknownModel(t *testing.T) {
	openai := &mockProvider{name: "openai", models: []string{"gpt-4o"}, resp: &Response{}}

	r := New(
		WithProvider("openai", openai),
	)

	_, err := r.Complete(t.Context(), &Request{Model: "does-not-exist"})
	if err == nil {
		t.Fatal("expected error for unknown model, got nil")
	}
	if !errors.Is(err, ErrUnknownModel) {
		t.Errorf("expected error to wrap ErrUnknownModel, got: %v", err)
	}
}

// --- Middleware ordering test ---

func TestMiddlewareOrder(t *testing.T) {
	// Each middleware prepends its tag to a shared log slice.
	// buildChain applies middlewares from last to first so that the first
	// declared middleware is outermost (called first).
	var callOrder []string

	makeMiddleware := func(tag string) MiddlewareFunc {
		return func(next Provider) Provider {
			return &orderTrackingProvider{
				inner: next,
				onCall: func() {
					callOrder = append(callOrder, tag)
				},
			}
		}
	}

	inner := &mockProvider{name: "inner", models: nil, resp: &Response{}}

	r := New(
		WithProvider("inner", inner),
		WithModelMapping("m", "inner"),
		WithMiddleware(
			makeMiddleware("first"),
			makeMiddleware("second"),
		),
	)

	_, err := r.Complete(t.Context(), &Request{Model: "m"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(callOrder) != 2 {
		t.Fatalf("expected 2 middleware calls, got %d: %v", len(callOrder), callOrder)
	}
	if callOrder[0] != "first" || callOrder[1] != "second" {
		t.Errorf("expected call order [first second], got %v", callOrder)
	}
}

// orderTrackingProvider is a Provider that calls onCall before delegating.
type orderTrackingProvider struct {
	inner  Provider
	onCall func()
}

func (o *orderTrackingProvider) Name() string     { return o.inner.Name() }
func (o *orderTrackingProvider) Models() []string { return o.inner.Models() }

func (o *orderTrackingProvider) Complete(ctx context.Context, req *Request) (*Response, error) {
	o.onCall()
	return o.inner.Complete(ctx, req)
}

func (o *orderTrackingProvider) Stream(ctx context.Context, req *Request) (*StreamResult, error) {
	o.onCall()
	return o.inner.Stream(ctx, req)
}

// --- Router.Close tests ---

type closableProvider struct {
	mockProvider
	closed bool
}

func (c *closableProvider) Close() error {
	c.closed = true
	return nil
}

func TestRouterClose_CallsCloser(t *testing.T) {
	closable := &closableProvider{mockProvider: mockProvider{name: "closable", models: nil}}
	plain := &mockProvider{name: "plain", models: nil}

	r := New(
		WithProvider("closable", closable),
		WithProvider("plain", plain),
	)

	if err := r.Close(); err != nil {
		t.Fatalf("unexpected error from Close: %v", err)
	}

	if !closable.closed {
		t.Error("expected closable provider to have Close() called")
	}
}
