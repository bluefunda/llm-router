package llmrouter

import (
	"errors"
	"testing"
)

func TestEloPolicy_UnseenModelsDefaultRating(t *testing.T) {
	p := NewEloPolicy()
	if got := p.Rating("unseen", ""); got != DefaultElo {
		t.Errorf("expected DefaultElo, got %v", got)
	}
}

func TestEloPolicy_SelectModel_PicksHighestRated(t *testing.T) {
	p := NewEloPolicy()
	candidates := []ModelConfig{
		{Provider: "a", Model: "model-a"},
		{Provider: "b", Model: "model-b"},
	}

	// Boost model-b above the default rating.
	for range 5 {
		p.ReportOutcome(EloOutcome{Model: "model-b", Success: true})
	}

	got, err := p.SelectModel(t.Context(), RoutingQuery{}, candidates)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Model != "model-b" {
		t.Errorf("expected model-b after positive outcomes, got %s", got.Model)
	}
}

func TestEloPolicy_SelectModel_TieBreaksToFirst(t *testing.T) {
	p := NewEloPolicy()
	candidates := []ModelConfig{
		{Provider: "a", Model: "model-a"},
		{Provider: "b", Model: "model-b"},
	}

	got, err := p.SelectModel(t.Context(), RoutingQuery{}, candidates)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Model != "model-a" {
		t.Errorf("expected model-a on tie, got %s", got.Model)
	}
}

func TestEloPolicy_SelectModel_NoCandidates(t *testing.T) {
	p := NewEloPolicy()
	_, err := p.SelectModel(t.Context(), RoutingQuery{}, nil)
	if !errors.Is(err, ErrNoCandidates) {
		t.Errorf("expected ErrNoCandidates, got %v", err)
	}
}

func TestEloPolicy_ReportOutcome_SuccessRaisesRating(t *testing.T) {
	p := NewEloPolicy()
	before := p.Rating("model-a", "")
	p.ReportOutcome(EloOutcome{Model: "model-a", Success: true})
	after := p.Rating("model-a", "")
	if after <= before {
		t.Errorf("expected rating to increase after success, before=%v after=%v", before, after)
	}
}

func TestEloPolicy_ReportOutcome_FailureLowersRating(t *testing.T) {
	p := NewEloPolicy()
	before := p.Rating("model-a", "")
	p.ReportOutcome(EloOutcome{Model: "model-a", Success: false})
	after := p.Rating("model-a", "")
	if after >= before {
		t.Errorf("expected rating to decrease after failure, before=%v after=%v", before, after)
	}
}

func TestEloPolicy_ReportOutcome_ScoreOverridesSuccess(t *testing.T) {
	p := NewEloPolicy()
	// Score 0.1 with Success true should still behave as a mostly-negative outcome.
	p.ReportOutcome(EloOutcome{Model: "model-a", Success: true, Score: 0.1})
	after := p.Rating("model-a", "")
	if after >= DefaultElo {
		t.Errorf("expected low Score to pull rating down despite Success=true, got %v", after)
	}
}

func TestEloPolicy_CategoriesAreIndependent(t *testing.T) {
	p := NewEloPolicy(WithEloCategoryFunc(func(q RoutingQuery) string { return q.Metadata["category"].(string) }))

	p.ReportOutcome(EloOutcome{Model: "model-a", Category: "code", Success: true})

	codeRating := p.Rating("model-a", "code")
	chatRating := p.Rating("model-a", "chat")
	if codeRating == chatRating {
		t.Errorf("expected independent ratings per category, code=%v chat=%v", codeRating, chatRating)
	}

	candidates := []ModelConfig{{Provider: "a", Model: "model-a"}, {Provider: "b", Model: "model-b"}}
	got, err := p.SelectModel(t.Context(), RoutingQuery{Metadata: map[string]any{"category": "code"}}, candidates)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Model != "model-a" {
		t.Errorf("expected model-a for boosted category, got %s", got.Model)
	}
}

func TestEloPolicy_KFactorControlsMagnitude(t *testing.T) {
	small := NewEloPolicy(WithEloKFactor(1))
	large := NewEloPolicy(WithEloKFactor(64))

	small.ReportOutcome(EloOutcome{Model: "m", Success: true})
	large.ReportOutcome(EloOutcome{Model: "m", Success: true})

	smallDelta := small.Rating("m", "") - DefaultElo
	largeDelta := large.Rating("m", "") - DefaultElo
	if largeDelta <= smallDelta {
		t.Errorf("expected larger K-factor to produce a bigger rating change, small=%v large=%v", smallDelta, largeDelta)
	}
}
