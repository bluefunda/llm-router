package main

import (
	"context"
	"fmt"
	"time"

	llmrouter "github.com/bluefunda/llm-router"
	"github.com/bluefunda/llm-router/middleware"
	"github.com/bluefunda/llm-router/providers/anthropic"
	"github.com/bluefunda/llm-router/providers/gemini"
	"github.com/bluefunda/llm-router/providers/openai"
)

func main() {
	ctx := context.Background()

	// Initialize Gemini (requires context for client creation)
	geminiProvider, err := gemini.NewFromEnv(ctx)
	if err != nil {
		fmt.Printf("Warning: Could not initialize Gemini: %v\n", err)
	}

	// Create router with multiple providers and middleware
	opts := []llmrouter.Option{
		llmrouter.WithProvider("openai", openai.NewFromEnv("openai", "OPENAI_API_KEY")),
		llmrouter.WithProvider("anthropic", anthropic.NewFromEnv()),
		llmrouter.WithProvider("deepseek", openai.NewFromEnv("deepseek", "DEEPSEEK_API_KEY")),
		llmrouter.WithProvider("groq", openai.NewFromEnv("groq", "GROQ_API_KEY")),

		// Model mappings - route specific models to providers
		llmrouter.WithModelMapping("gpt-4o", "openai"),
		llmrouter.WithModelMapping("gpt-4o-mini", "openai"),
		llmrouter.WithModelMapping("claude-sonnet-4-20250514", "anthropic"),
		llmrouter.WithModelMapping("deepseek-chat", "deepseek"),
		llmrouter.WithModelMapping("llama-3.3-70b-versatile", "groq"),

		// Middleware
		llmrouter.WithMiddleware(
			middleware.NewRetryMiddleware(3, time.Second),
			middleware.NewCircuitBreakerMiddleware("llm-router", 5, 30*time.Second),
			middleware.NewTimeoutMiddleware(60*time.Second),
		),
	}

	// Add Gemini if available
	if geminiProvider != nil {
		opts = append(opts,
			llmrouter.WithProvider("gemini", geminiProvider),
			llmrouter.WithModelMapping("gemini-1.5-flash", "gemini"),
		)
	}

	router := llmrouter.New(opts...)

	fmt.Println("Registered providers:", router.Providers())
	fmt.Println()

	// Test different providers
	testCases := []struct {
		name  string
		model string
	}{
		{"OpenAI", "gpt-4o-mini"},
		{"Anthropic", "claude-sonnet-4-20250514"},
		{"DeepSeek", "deepseek-chat"},
		{"Groq", "llama-3.3-70b-versatile"},
	}

	if geminiProvider != nil {
		testCases = append(testCases, struct {
			name  string
			model string
		}{"Gemini", "gemini-1.5-flash"})
	}

	for _, tc := range testCases {
		fmt.Printf("--- %s (%s) ---\n", tc.name, tc.model)

		resp, err := router.Complete(ctx, &llmrouter.Request{
			Model: tc.model,
			Messages: []llmrouter.Message{
				{Role: llmrouter.RoleUser, Content: "Say 'Hello from ' followed by your model name in 10 words or less."},
			},
			MaxTokens: intPtr(50),
		})

		if err != nil {
			fmt.Printf("Error: %v\n\n", err)
			continue
		}

		fmt.Printf("Provider: %s\n", resp.Provider)
		fmt.Printf("Response: %s\n", resp.Choices[0].Message.Content)
		if resp.Usage != nil {
			fmt.Printf("Tokens: %d total\n", resp.Usage.TotalTokens)
		}
		fmt.Println()
	}

	// Clean up Gemini client
	if geminiProvider != nil {
		geminiProvider.Close()
	}
}

func intPtr(i int) *int {
	return &i
}
