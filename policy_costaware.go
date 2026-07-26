package llmrouter

import (
	"context"
	"fmt"
	"sort"
)

// ComplexityFunc scores a RoutingQuery's complexity. Higher scores indicate
// a more demanding request that needs a more capable model.
type ComplexityFunc func(query RoutingQuery) float64

// EstimateComplexity is the default ComplexityFunc: a rough token-count
// estimate (~4 characters per token) across all message content.
func EstimateComplexity(query RoutingQuery) float64 {
	var chars int
	for _, m := range query.Messages {
		chars += len(m.Content)
		for _, part := range m.ContentParts {
			chars += len(part.Text)
		}
	}
	return float64(chars) / 4
}

// ComplexityTier maps a complexity threshold to the minimum candidate Tier
// required to handle it. CostAwarePolicy applies the first tier (in
// ascending MaxComplexity order) whose MaxComplexity the query's complexity
// does not exceed; if it exceeds all of them, the highest tier's MinTier applies.
type ComplexityTier struct {
	MaxComplexity float64
	MinTier       int
}

// minTierMetadataKey is the RoutingQuery.Metadata key CostAwarePolicy checks
// for a caller-supplied minimum-tier override.
const minTierMetadataKey = "llmrouter_min_tier"

// EscalateMetadata returns a Request.Metadata fragment that tells
// CostAwarePolicy to require at least minTier on this request, overriding
// whatever its complexity scoring would otherwise choose. Callers merge
// this in and resubmit after observing an error or a low-confidence
// response, to force a stronger model on the next attempt:
//
//	req.Metadata = llmrouter.EscalateMetadata(previousTier + 1)
//	resp, err := router.Complete(ctx, req)
func EscalateMetadata(minTier int) map[string]any {
	return map[string]any{minTierMetadataKey: minTier}
}

// CostAwarePolicy selects the cheapest candidate model (by ModelConfig.CostHint)
// whose ModelConfig.Tier meets the minimum required by query complexity, or by
// an explicit override from EscalateMetadata. Candidates with no registered
// [ModelConfig] metadata (via WithModelConfig) are treated as Tier 0 /
// CostHint 0 — register metadata for the models you want ranked meaningfully.
//
// If no candidate meets the minimum tier, SelectModel returns an error
// wrapping ErrNoCandidates. Compose with [PolicyChain] to fall back to
// another policy in that case, e.g. PolicyChain{CostAwarePolicy{...}, StaticPolicy{}}.
type CostAwarePolicy struct {
	// Complexity scores a query; defaults to EstimateComplexity if nil.
	Complexity ComplexityFunc
	// Tiers maps complexity to a minimum required Tier.
	Tiers []ComplexityTier
}

// SelectModel implements RoutingPolicy.
func (p CostAwarePolicy) SelectModel(_ context.Context, query RoutingQuery, candidates []ModelConfig) (ModelConfig, error) {
	if len(candidates) == 0 {
		return ModelConfig{}, ErrNoCandidates
	}

	minTier := p.requiredTier(query)

	eligible := make([]ModelConfig, 0, len(candidates))
	for _, c := range candidates {
		if c.Tier >= minTier {
			eligible = append(eligible, c)
		}
	}
	if len(eligible) == 0 {
		return ModelConfig{}, fmt.Errorf("%w: no candidate meets minimum tier %d", ErrNoCandidates, minTier)
	}

	sort.SliceStable(eligible, func(i, j int) bool {
		return eligible[i].CostHint < eligible[j].CostHint
	})
	return eligible[0], nil
}

// requiredTier resolves the minimum tier for a query: an explicit
// EscalateMetadata override takes precedence over complexity scoring.
func (p CostAwarePolicy) requiredTier(query RoutingQuery) int {
	if v, ok := query.Metadata[minTierMetadataKey]; ok {
		if tier, ok := toInt(v); ok {
			return tier
		}
	}

	complexityFn := p.Complexity
	if complexityFn == nil {
		complexityFn = EstimateComplexity
	}
	score := complexityFn(query)

	tiers := make([]ComplexityTier, len(p.Tiers))
	copy(tiers, p.Tiers)
	sort.Slice(tiers, func(i, j int) bool { return tiers[i].MaxComplexity < tiers[j].MaxComplexity })

	for _, t := range tiers {
		if score <= t.MaxComplexity {
			return t.MinTier
		}
	}
	if len(tiers) > 0 {
		return tiers[len(tiers)-1].MinTier
	}
	return 0
}

func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case float64:
		return int(n), true
	default:
		return 0, false
	}
}
