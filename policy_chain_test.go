package llmrouter

import (
	"context"
	"errors"
	"testing"
)

func TestPolicyChain_FirstSuccessWins(t *testing.T) {
	candidates := []ModelConfig{{Provider: "a", Model: "model-a"}}
	failing := policyFunc(func(context.Context, RoutingQuery, []ModelConfig) (ModelConfig, error) {
		return ModelConfig{}, errors.New("nope")
	})

	chain := PolicyChain{failing, StaticPolicy{}}
	got, err := chain.SelectModel(t.Context(), RoutingQuery{}, candidates)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Model != "model-a" {
		t.Errorf("expected model-a via second policy, got %s", got.Model)
	}
}

func TestPolicyChain_NilEntriesSkipped(t *testing.T) {
	candidates := []ModelConfig{{Provider: "a", Model: "model-a"}}
	chain := PolicyChain{nil, StaticPolicy{}}
	got, err := chain.SelectModel(t.Context(), RoutingQuery{}, candidates)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Model != "model-a" {
		t.Errorf("expected model-a, got %s", got.Model)
	}
}

func TestPolicyChain_AllFail_ReturnsLastError(t *testing.T) {
	wantErr := errors.New("second failure")
	first := policyFunc(func(context.Context, RoutingQuery, []ModelConfig) (ModelConfig, error) {
		return ModelConfig{}, errors.New("first failure")
	})
	second := policyFunc(func(context.Context, RoutingQuery, []ModelConfig) (ModelConfig, error) {
		return ModelConfig{}, wantErr
	})

	chain := PolicyChain{first, second}
	_, err := chain.SelectModel(t.Context(), RoutingQuery{}, []ModelConfig{{Provider: "a", Model: "model-a"}})
	if !errors.Is(err, wantErr) {
		t.Errorf("expected last policy's error, got %v", err)
	}
}

func TestPolicyChain_Empty(t *testing.T) {
	chain := PolicyChain{}
	_, err := chain.SelectModel(t.Context(), RoutingQuery{}, []ModelConfig{{Provider: "a", Model: "model-a"}})
	if !errors.Is(err, ErrNoCandidates) {
		t.Errorf("expected ErrNoCandidates for empty chain, got %v", err)
	}
}
