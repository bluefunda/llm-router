package llmrouter

import (
	"math"
	"testing"
)

func TestCalculateCost_KnownModel(t *testing.T) {
	usage := &Usage{PromptTokens: 1000, CompletionTokens: 500, TotalTokens: 1500}
	cost := CalculateCost("gpt-4o", usage, DefaultPrices)
	// 1000 * 2.50/1M + 500 * 10.00/1M = 0.0025 + 0.005 = 0.0075
	want := 0.0075
	if math.Abs(cost-want) > 1e-9 {
		t.Errorf("expected %v, got %v", want, cost)
	}
}

func TestCalculateCost_CachedTokens(t *testing.T) {
	usage := &Usage{
		PromptTokens:       1000,
		CachedPromptTokens: 800,
		CompletionTokens:   200,
		TotalTokens:        1200,
	}
	cost := CalculateCost("gpt-4o", usage, DefaultPrices)
	// uncached: 200 * 2.50/1M = 0.0005
	// cached:   800 * 1.25/1M = 0.001
	// output:   200 * 10.00/1M = 0.002
	// total = 0.0035
	want := 0.0035
	if math.Abs(cost-want) > 1e-9 {
		t.Errorf("expected %v, got %v", want, cost)
	}
}

func TestCalculateCost_UnknownModel(t *testing.T) {
	usage := &Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150}
	cost := CalculateCost("unknown-model-xyz", usage, DefaultPrices)
	if cost != 0 {
		t.Errorf("expected 0 for unknown model, got %v", cost)
	}
}

func TestCalculateCost_NilUsage(t *testing.T) {
	cost := CalculateCost("gpt-4o", nil, DefaultPrices)
	if cost != 0 {
		t.Errorf("expected 0 for nil usage, got %v", cost)
	}
}

func TestCalculateCost_EmptyPrices(t *testing.T) {
	usage := &Usage{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150}
	cost := CalculateCost("gpt-4o", usage, map[string]ModelPrice{})
	if cost != 0 {
		t.Errorf("expected 0 for empty price table, got %v", cost)
	}
}

func TestCalculateCost_CustomPriceTable(t *testing.T) {
	prices := map[string]ModelPrice{
		"custom-model": {InputPerMillion: 1.00, OutputPerMillion: 2.00},
	}
	usage := &Usage{PromptTokens: 1_000_000, CompletionTokens: 1_000_000, TotalTokens: 2_000_000}
	cost := CalculateCost("custom-model", usage, prices)
	if math.Abs(cost-3.00) > 1e-9 {
		t.Errorf("expected 3.00, got %v", cost)
	}
}

func TestCalculateCost_AnthropicModel(t *testing.T) {
	usage := &Usage{
		PromptTokens:       500,
		CachedPromptTokens: 400,
		CompletionTokens:   100,
		TotalTokens:        600,
	}
	cost := CalculateCost("claude-sonnet-4-20250514", usage, DefaultPrices)
	// uncached: 100 * 3.00/1M = 0.0003
	// cached:   400 * 0.30/1M = 0.00012
	// output:   100 * 15.00/1M = 0.0015
	// total = 0.00192
	want := 0.00192
	if math.Abs(cost-want) > 1e-9 {
		t.Errorf("expected %v, got %v", want, cost)
	}
}

func TestUsage_CacheHitRate(t *testing.T) {
	u := &Usage{PromptTokens: 1000, CachedPromptTokens: 400}
	got := u.CacheHitRate()
	want := 0.4
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("expected %v, got %v", want, got)
	}
}

func TestUsage_CacheHitRate_NoTokens(t *testing.T) {
	u := &Usage{}
	if u.CacheHitRate() != 0 {
		t.Error("expected 0 for zero prompt tokens")
	}
}

func TestUsage_CacheHitRate_NilReceiver(t *testing.T) {
	var u *Usage
	if u.CacheHitRate() != 0 {
		t.Error("expected 0 for nil usage")
	}
}

func TestDefaultPrices_CoveredModels(t *testing.T) {
	required := []string{
		"gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini",
		"claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022",
		"gemini-2.5-pro", "gemini-2.5-flash",
	}
	for _, model := range required {
		if _, ok := DefaultPrices[model]; !ok {
			t.Errorf("DefaultPrices missing entry for %q", model)
		}
	}
}
