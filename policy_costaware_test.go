package llmrouter

import (
	"errors"
	"testing"
)

func TestCostAwarePolicy_PicksCheapestMeetingTier(t *testing.T) {
	candidates := []ModelConfig{
		{Provider: "a", Model: "expensive-tier2", Tier: 2, CostHint: 10},
		{Provider: "b", Model: "cheap-tier2", Tier: 2, CostHint: 1},
		{Provider: "c", Model: "cheapest-tier1", Tier: 1, CostHint: 0.1},
	}
	policy := CostAwarePolicy{
		Tiers: []ComplexityTier{{MaxComplexity: 1e9, MinTier: 2}},
	}

	got, err := policy.SelectModel(t.Context(), RoutingQuery{}, candidates)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Model != "cheap-tier2" {
		t.Errorf("expected cheap-tier2, got %s", got.Model)
	}
}

func TestCostAwarePolicy_ComplexityDrivesTierThreshold(t *testing.T) {
	candidates := []ModelConfig{
		{Provider: "a", Model: "small", Tier: 1, CostHint: 0.1},
		{Provider: "b", Model: "large", Tier: 2, CostHint: 5},
	}
	policy := CostAwarePolicy{
		Complexity: func(RoutingQuery) float64 { return 100 },
		Tiers: []ComplexityTier{
			{MaxComplexity: 50, MinTier: 1},
			{MaxComplexity: 1000, MinTier: 2},
		},
	}

	got, err := policy.SelectModel(t.Context(), RoutingQuery{}, candidates)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Model != "large" {
		t.Errorf("expected large (tier 2 required at complexity 100), got %s", got.Model)
	}
}

func TestCostAwarePolicy_EscalateMetadataOverridesComplexity(t *testing.T) {
	candidates := []ModelConfig{
		{Provider: "a", Model: "small", Tier: 1, CostHint: 0.1},
		{Provider: "b", Model: "large", Tier: 2, CostHint: 5},
	}
	policy := CostAwarePolicy{
		Complexity: func(RoutingQuery) float64 { return 0 }, // would otherwise pick tier 0
	}

	query := RoutingQuery{Metadata: EscalateMetadata(2)}
	got, err := policy.SelectModel(t.Context(), query, candidates)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.Model != "large" {
		t.Errorf("expected escalation to large, got %s", got.Model)
	}
}

func TestCostAwarePolicy_NoCandidateMeetsTier(t *testing.T) {
	candidates := []ModelConfig{{Provider: "a", Model: "small", Tier: 1}}
	policy := CostAwarePolicy{Tiers: []ComplexityTier{{MaxComplexity: 1e9, MinTier: 5}}}

	_, err := policy.SelectModel(t.Context(), RoutingQuery{}, candidates)
	if !errors.Is(err, ErrNoCandidates) {
		t.Errorf("expected ErrNoCandidates, got %v", err)
	}
}

func TestCostAwarePolicy_NoCandidates(t *testing.T) {
	policy := CostAwarePolicy{}
	_, err := policy.SelectModel(t.Context(), RoutingQuery{}, nil)
	if !errors.Is(err, ErrNoCandidates) {
		t.Errorf("expected ErrNoCandidates, got %v", err)
	}
}

func TestCostAwarePolicy_ChainedWithStaticFallback(t *testing.T) {
	candidates := []ModelConfig{{Provider: "a", Model: "small", Tier: 1}}
	chain := PolicyChain{
		CostAwarePolicy{Tiers: []ComplexityTier{{MaxComplexity: 1e9, MinTier: 5}}}, // no candidate meets tier 5
		StaticPolicy{},
	}

	got, err := chain.SelectModel(t.Context(), RoutingQuery{}, candidates)
	if err != nil {
		t.Fatalf("expected fallback to StaticPolicy to succeed, got: %v", err)
	}
	if got.Model != "small" {
		t.Errorf("expected small via StaticPolicy fallback, got %s", got.Model)
	}
}

func TestEstimateComplexity(t *testing.T) {
	tests := []struct {
		name     string
		query    RoutingQuery
		wantChar float64
	}{
		{
			name:     "empty",
			query:    RoutingQuery{},
			wantChar: 0,
		},
		{
			name: "single message content",
			query: RoutingQuery{
				Messages: []Message{{Content: "12345678"}}, // 8 chars
			},
			wantChar: 2, // 8 / 4
		},
		{
			name: "content parts included",
			query: RoutingQuery{
				Messages: []Message{{ContentParts: []ContentPart{{Text: "1234"}}}},
			},
			wantChar: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EstimateComplexity(tt.query)
			if got != tt.wantChar {
				t.Errorf("expected %v, got %v", tt.wantChar, got)
			}
		})
	}
}
