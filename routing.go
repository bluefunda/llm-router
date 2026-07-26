package llmrouter

import (
	"context"
	"fmt"
)

// ModelConfig is a routing candidate: a specific model available on a
// specific registered provider, with optional metadata a RoutingPolicy can
// use to make a selection.
type ModelConfig struct {
	Provider string  // registered provider name
	Model    string  // model ID as accepted by the provider
	Tier     int     // relative capability tier; higher means more capable. 0 if unset.
	CostHint float64 // relative cost signal (e.g. USD per million tokens); 0 if unset
}

// RoutingQuery describes the request a RoutingPolicy is selecting a model for.
type RoutingQuery struct {
	Model    string         // model requested by the caller, may be empty
	Messages []Message      // request messages, useful for complexity scoring
	Metadata map[string]any // pass-through from Request.Metadata
}

// RoutingPolicy selects a model from a set of candidates for a given query.
// It is consulted by the Router in place of the built-in static resolution
// (see [WithRoutingPolicy] / [Router.SetRoutingPolicy]) when one is configured.
// Implementations must be safe for concurrent use.
type RoutingPolicy interface {
	SelectModel(ctx context.Context, query RoutingQuery, candidates []ModelConfig) (ModelConfig, error)
}

// ErrNoCandidates is returned by a RoutingPolicy when no candidate models are available.
var ErrNoCandidates = fmt.Errorf("no candidate models available")

// StaticPolicy is the reference RoutingPolicy implementation: it replicates
// the router's built-in static resolution over the candidate list — an exact
// model match if one was requested, otherwise the first candidate in
// registration order. Its zero value is ready to use.
//
// Note this differs subtly from the router's default (policy-less) path:
// StaticPolicy matches candidates by model ID only and does not consult
// [WithModelMapping] aliases, which are a router-internal lookup.
type StaticPolicy struct{}

// SelectModel implements RoutingPolicy.
func (StaticPolicy) SelectModel(_ context.Context, query RoutingQuery, candidates []ModelConfig) (ModelConfig, error) {
	if len(candidates) == 0 {
		return ModelConfig{}, ErrNoCandidates
	}
	if query.Model == "" {
		return candidates[0], nil
	}
	for _, c := range candidates {
		if c.Model == query.Model {
			return c, nil
		}
	}
	return ModelConfig{}, fmt.Errorf("%w: %s", ErrUnknownModel, query.Model)
}
