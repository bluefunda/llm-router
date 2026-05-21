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
	defer stream.Close()
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
