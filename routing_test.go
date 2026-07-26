package llmrouter

import (
	"context"
	"errors"
	"testing"
)

// recordingProvider records the last request it received, so tests can
// verify what model a RoutingPolicy actually caused to be sent downstream.
type recordingProvider struct {
	name    string
	models  []string
	resp    *Response
	lastReq *Request
}

func (p *recordingProvider) Name() string     { return p.name }
func (p *recordingProvider) Models() []string { return p.models }

func (p *recordingProvider) Complete(_ context.Context, req *Request) (*Response, error) {
	p.lastReq = req
	return p.resp, nil
}

func (p *recordingProvider) Stream(_ context.Context, req *Request) (*StreamResult, error) {
	p.lastReq = req
	ch := make(chan Event, 1)
	ch <- Event{Type: EventDone}
	close(ch)
	return NewStreamResult(ch), nil
}

// policyFunc adapts a function to RoutingPolicy.
type policyFunc func(ctx context.Context, q RoutingQuery, candidates []ModelConfig) (ModelConfig, error)

func (f policyFunc) SelectModel(ctx context.Context, q RoutingQuery, candidates []ModelConfig) (ModelConfig, error) {
	return f(ctx, q, candidates)
}

// --- StaticPolicy tests ---

func TestStaticPolicy_ExactMatch(t *testing.T) {
	candidates := []ModelConfig{
		{Provider: "a", Model: "model-a"},
		{Provider: "b", Model: "model-b"},
	}
	got, err := (StaticPolicy{}).SelectModel(t.Context(), RoutingQuery{Model: "model-b"}, candidates)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Provider != "b" {
		t.Errorf("expected provider b, got %s", got.Provider)
	}
}

func TestStaticPolicy_NoModelRequested_PicksFirst(t *testing.T) {
	candidates := []ModelConfig{
		{Provider: "a", Model: "model-a"},
		{Provider: "b", Model: "model-b"},
	}
	got, err := (StaticPolicy{}).SelectModel(t.Context(), RoutingQuery{}, candidates)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Provider != "a" {
		t.Errorf("expected first candidate (provider a), got %s", got.Provider)
	}
}

func TestStaticPolicy_UnknownModel(t *testing.T) {
	candidates := []ModelConfig{{Provider: "a", Model: "model-a"}}
	_, err := (StaticPolicy{}).SelectModel(t.Context(), RoutingQuery{Model: "does-not-exist"}, candidates)
	if !errors.Is(err, ErrUnknownModel) {
		t.Errorf("expected ErrUnknownModel, got %v", err)
	}
}

func TestStaticPolicy_NoCandidates(t *testing.T) {
	_, err := (StaticPolicy{}).SelectModel(t.Context(), RoutingQuery{}, nil)
	if !errors.Is(err, ErrNoCandidates) {
		t.Errorf("expected ErrNoCandidates, got %v", err)
	}
}

// --- Router integration tests ---

func TestRouter_NoPolicy_UsesStaticResolution(t *testing.T) {
	// Regression: with no policy configured, behavior must be identical to
	// the router's pre-RoutingPolicy resolution.
	a := &recordingProvider{name: "a", models: []string{"model-a"}, resp: &Response{Model: "model-a"}}

	r := New(
		WithProvider("a", a),
		WithModelMapping("model-a", "a"),
	)

	resp, err := r.Complete(t.Context(), &Request{Model: "model-a"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Model != "model-a" {
		t.Errorf("expected model-a, got %s", resp.Model)
	}
	if a.lastReq.Model != "model-a" {
		t.Errorf("expected downstream request model model-a, got %s", a.lastReq.Model)
	}
}

func TestRouter_WithPolicy_OverridesModelSelection(t *testing.T) {
	a := &recordingProvider{name: "a", models: []string{"model-a"}, resp: &Response{Model: "model-a"}}
	b := &recordingProvider{name: "b", models: []string{"model-b"}, resp: &Response{Model: "model-b"}}

	// Policy always picks provider b's model regardless of what was requested.
	policy := policyFunc(func(_ context.Context, _ RoutingQuery, candidates []ModelConfig) (ModelConfig, error) {
		for _, c := range candidates {
			if c.Provider == "b" {
				return c, nil
			}
		}
		return ModelConfig{}, ErrNoCandidates
	})

	r := New(
		WithProvider("a", a),
		WithProvider("b", b),
		WithRoutingPolicy(policy),
	)

	resp, err := r.Complete(t.Context(), &Request{Model: "model-a"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Model != "model-b" {
		t.Errorf("expected policy-selected model-b, got %s", resp.Model)
	}
	if a.lastReq != nil {
		t.Error("expected provider a to never be called")
	}
	if b.lastReq == nil || b.lastReq.Model != "model-b" {
		t.Errorf("expected downstream request model model-b, got %+v", b.lastReq)
	}
}

func TestRouter_WithPolicy_Streaming(t *testing.T) {
	b := &recordingProvider{name: "b", models: []string{"model-b"}}

	policy := policyFunc(func(_ context.Context, _ RoutingQuery, candidates []ModelConfig) (ModelConfig, error) {
		return candidates[0], nil
	})

	r := New(
		WithProvider("b", b),
		WithRoutingPolicy(policy),
	)

	stream, err := r.Stream(t.Context(), &Request{Model: "anything"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer stream.Close() //nolint:errcheck
	for stream.Next() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
	if b.lastReq == nil || b.lastReq.Model != "model-b" {
		t.Errorf("expected downstream request model model-b, got %+v", b.lastReq)
	}
}

func TestRouter_WithPolicy_UnknownProviderSelected(t *testing.T) {
	a := &recordingProvider{name: "a", models: []string{"model-a"}}

	policy := policyFunc(func(_ context.Context, _ RoutingQuery, _ []ModelConfig) (ModelConfig, error) {
		return ModelConfig{Provider: "ghost", Model: "model-a"}, nil
	})

	r := New(
		WithProvider("a", a),
		WithRoutingPolicy(policy),
	)

	_, err := r.Complete(t.Context(), &Request{Model: "model-a"})
	if !errors.Is(err, ErrUnknownProvider) {
		t.Errorf("expected ErrUnknownProvider, got %v", err)
	}
}

func TestRouter_WithPolicy_PropagatesPolicyError(t *testing.T) {
	a := &recordingProvider{name: "a", models: []string{"model-a"}}
	wantErr := errors.New("policy refused")

	policy := policyFunc(func(_ context.Context, _ RoutingQuery, _ []ModelConfig) (ModelConfig, error) {
		return ModelConfig{}, wantErr
	})

	r := New(
		WithProvider("a", a),
		WithRoutingPolicy(policy),
	)

	_, err := r.Complete(t.Context(), &Request{Model: "model-a"})
	if !errors.Is(err, wantErr) {
		t.Errorf("expected policy error to propagate, got %v", err)
	}
}

func TestRouter_ModelConfig_PopulatesCandidateMetadata(t *testing.T) {
	a := &recordingProvider{name: "a", models: []string{"model-a"}, resp: &Response{Model: "model-a"}}

	var seen []ModelConfig
	policy := policyFunc(func(_ context.Context, _ RoutingQuery, candidates []ModelConfig) (ModelConfig, error) {
		seen = candidates
		return candidates[0], nil
	})

	r := New(
		WithProvider("a", a),
		WithRoutingPolicy(policy),
		WithModelConfig(ModelConfig{Model: "model-a", Tier: 2, CostHint: 1.5}),
	)

	_, err := r.Complete(t.Context(), &Request{Model: "model-a"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(seen) != 1 {
		t.Fatalf("expected 1 candidate, got %d", len(seen))
	}
	if seen[0].Tier != 2 || seen[0].CostHint != 1.5 {
		t.Errorf("expected metadata Tier=2 CostHint=1.5, got %+v", seen[0])
	}
}

func TestRouter_SetRoutingPolicy_RuntimeUpdate(t *testing.T) {
	a := &recordingProvider{name: "a", models: []string{"model-a"}, resp: &Response{Model: "model-a"}}
	r := New(WithProvider("a", a), WithModelMapping("model-a", "a"))

	called := false
	r.SetRoutingPolicy(policyFunc(func(_ context.Context, _ RoutingQuery, candidates []ModelConfig) (ModelConfig, error) {
		called = true
		return candidates[0], nil
	}))

	_, err := r.Complete(t.Context(), &Request{Model: "model-a"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Error("expected policy set via SetRoutingPolicy to be consulted")
	}
}
