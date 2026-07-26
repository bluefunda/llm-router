package llmrouter

import "context"

// PolicyChain tries each policy in order and returns the first successful
// selection, so specialized policies can be composed with a safe default —
// e.g. PolicyChain{CostAwarePolicy{...}, StaticPolicy{}} falls back to
// static resolution if the cost-aware policy finds no eligible candidate.
// A nil entry is skipped.
type PolicyChain []RoutingPolicy

// SelectModel implements RoutingPolicy.
func (c PolicyChain) SelectModel(ctx context.Context, query RoutingQuery, candidates []ModelConfig) (ModelConfig, error) {
	var lastErr error
	for _, p := range c {
		if p == nil {
			continue
		}
		selected, err := p.SelectModel(ctx, query, candidates)
		if err == nil {
			return selected, nil
		}
		lastErr = err
	}
	if lastErr == nil {
		lastErr = ErrNoCandidates
	}
	return ModelConfig{}, lastErr
}
