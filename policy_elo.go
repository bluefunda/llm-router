package llmrouter

import (
	"context"
	"math"
	"sync"
)

// DefaultElo is the starting rating assigned to any (model, category) pair
// that has never had an outcome reported.
const DefaultElo = 1200.0

// EloOutcome is the result of a completed request, reported via
// [EloPolicy.ReportOutcome] to update that model's rating for future
// routing decisions.
type EloOutcome struct {
	Model    string  // model ID, matching ModelConfig.Model
	Category string  // task category bucket; "" is a valid default bucket
	Success  bool    // whether the response was usable
	Score    float64 // optional finer-grained quality signal in [0,1]; if 0, Success alone drives the update
}

type eloKey struct {
	model    string
	category string
}

// EloOption configures an EloPolicy constructed via NewEloPolicy.
type EloOption func(*EloPolicy)

// WithEloKFactor sets the Elo K-factor (rating change magnitude per
// reported outcome). Defaults to 32.
func WithEloKFactor(k float64) EloOption {
	return func(p *EloPolicy) { p.kFactor = k }
}

// WithEloCategoryFunc sets the function used to bucket a RoutingQuery into a
// task category for rating lookups. Defaults to a single "" category for
// all queries.
func WithEloCategoryFunc(f func(RoutingQuery) string) EloOption {
	return func(p *EloPolicy) { p.category = f }
}

// EloPolicy biases model selection using a lightweight Elo-style rating
// maintained per (model, category), entirely in memory. Ratings start at
// DefaultElo and move via [EloPolicy.ReportOutcome], which callers invoke
// after each request completes (success/failure, or a finer-grained quality
// score). SelectModel picks the highest-rated eligible candidate. Safe for
// concurrent use.
type EloPolicy struct {
	mu       sync.RWMutex
	ratings  map[eloKey]float64
	kFactor  float64
	category func(RoutingQuery) string
}

// NewEloPolicy creates an EloPolicy ready to use.
func NewEloPolicy(opts ...EloOption) *EloPolicy {
	p := &EloPolicy{
		ratings: make(map[eloKey]float64),
		kFactor: 32,
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

// SelectModel implements RoutingPolicy: it returns the candidate with the
// highest rating in the query's category, defaulting unseen models to
// DefaultElo. Ties keep the first candidate in registration order.
func (p *EloPolicy) SelectModel(_ context.Context, query RoutingQuery, candidates []ModelConfig) (ModelConfig, error) {
	if len(candidates) == 0 {
		return ModelConfig{}, ErrNoCandidates
	}

	category := ""
	if p.category != nil {
		category = p.category(query)
	}

	p.mu.RLock()
	defer p.mu.RUnlock()

	best := candidates[0]
	bestRating := p.ratingLocked(best.Model, category)
	for _, c := range candidates[1:] {
		r := p.ratingLocked(c.Model, category)
		if r > bestRating {
			best, bestRating = c, r
		}
	}
	return best, nil
}

// Rating returns the current rating for a (model, category) pair, or
// DefaultElo if no outcome has been reported for it yet.
func (p *EloPolicy) Rating(model, category string) float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.ratingLocked(model, category)
}

func (p *EloPolicy) ratingLocked(model, category string) float64 {
	if r, ok := p.ratings[eloKey{model: model, category: category}]; ok {
		return r
	}
	return DefaultElo
}

// ReportOutcome updates a model's rating from a completed request's
// outcome. Call this after each request — typically from a
// [github.com/bluefunda/llmrouter/middleware.ResponseHook] — to feed real
// success/failure or quality signal back into future SelectModel calls.
func (p *EloPolicy) ReportOutcome(outcome EloOutcome) {
	actual := 0.0
	switch {
	case outcome.Score != 0:
		actual = math.Max(0, math.Min(1, outcome.Score))
	case outcome.Success:
		actual = 1
	}

	key := eloKey{model: outcome.Model, category: outcome.Category}

	p.mu.Lock()
	defer p.mu.Unlock()
	rating, ok := p.ratings[key]
	if !ok {
		rating = DefaultElo
	}
	expected := 1 / (1 + math.Pow(10, (DefaultElo-rating)/400))
	p.ratings[key] = rating + p.kFactor*(actual-expected)
}
