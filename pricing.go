package llmrouter

// ModelPrice holds the per-token USD pricing for a model.
type ModelPrice struct {
	InputPerMillion     float64 // USD per million input (prompt) tokens
	OutputPerMillion    float64 // USD per million output (completion) tokens
	CacheReadPerMillion float64 // USD per million cache-read tokens; 0 if not applicable
}

// DefaultPrices is the built-in price table for known models.
// Cost is 0 for models not present in the map.
// Prices reflect standard API rates as of mid-2025; override with WithPriceTable if needed.
var DefaultPrices = map[string]ModelPrice{
	// OpenAI
	"gpt-4.1":      {InputPerMillion: 2.00, OutputPerMillion: 8.00, CacheReadPerMillion: 0.50},
	"gpt-4.1-mini": {InputPerMillion: 0.40, OutputPerMillion: 1.60, CacheReadPerMillion: 0.10},
	"gpt-4.1-nano": {InputPerMillion: 0.10, OutputPerMillion: 0.40, CacheReadPerMillion: 0.025},
	"gpt-4o":       {InputPerMillion: 2.50, OutputPerMillion: 10.00, CacheReadPerMillion: 1.25},
	"gpt-4o-mini":  {InputPerMillion: 0.15, OutputPerMillion: 0.60, CacheReadPerMillion: 0.075},
	"o4-mini":      {InputPerMillion: 1.10, OutputPerMillion: 4.40, CacheReadPerMillion: 0.275},

	// Anthropic Claude
	"claude-opus-4-20250514":    {InputPerMillion: 15.00, OutputPerMillion: 75.00, CacheReadPerMillion: 1.50},
	"claude-sonnet-4-20250514":  {InputPerMillion: 3.00, OutputPerMillion: 15.00, CacheReadPerMillion: 0.30},
	"claude-3-5-haiku-20241022": {InputPerMillion: 0.80, OutputPerMillion: 4.00, CacheReadPerMillion: 0.08},
	"claude-3-5-sonnet-20241022": {InputPerMillion: 3.00, OutputPerMillion: 15.00, CacheReadPerMillion: 0.30},
	"claude-3-opus-20240229":    {InputPerMillion: 15.00, OutputPerMillion: 75.00, CacheReadPerMillion: 1.50},
	"claude-3-sonnet-20240229":  {InputPerMillion: 3.00, OutputPerMillion: 15.00, CacheReadPerMillion: 0.30},
	"claude-3-haiku-20240307":   {InputPerMillion: 0.25, OutputPerMillion: 1.25, CacheReadPerMillion: 0.03},

	// DeepSeek
	"deepseek-chat":  {InputPerMillion: 0.07, OutputPerMillion: 1.10},
	"deepseek-coder": {InputPerMillion: 0.07, OutputPerMillion: 1.10},

	// Google Gemini
	"gemini-2.5-pro":   {InputPerMillion: 1.25, OutputPerMillion: 10.00},
	"gemini-2.5-flash": {InputPerMillion: 0.15, OutputPerMillion: 0.60},
	"gemini-2.0-flash": {InputPerMillion: 0.10, OutputPerMillion: 0.40},
}

// CalculateCost returns the estimated USD cost for a request given token usage and a price table.
// Cached tokens are billed at CacheReadPerMillion; uncached prompt tokens at InputPerMillion.
// Returns 0 if the model is not in the price table or usage is nil.
func CalculateCost(model string, usage *Usage, prices map[string]ModelPrice) float64 {
	if usage == nil || len(prices) == 0 {
		return 0
	}
	price, ok := prices[model]
	if !ok {
		return 0
	}
	uncached := usage.PromptTokens - usage.CachedPromptTokens
	cost := float64(uncached) * price.InputPerMillion / 1_000_000
	cost += float64(usage.CachedPromptTokens) * price.CacheReadPerMillion / 1_000_000
	cost += float64(usage.CompletionTokens) * price.OutputPerMillion / 1_000_000
	return cost
}
