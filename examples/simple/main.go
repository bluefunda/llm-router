package main

import (
	"context"
	"fmt"
	"os"
	"time"

	llmrouter "github.com/bluefunda/llm-router"
	"github.com/bluefunda/llm-router/middleware"
	"github.com/bluefunda/llm-router/providers/anthropic"
	"github.com/bluefunda/llm-router/providers/openai"
)

func main() {
	ctx := context.Background()

	// Create router with multiple providers
	router := llmrouter.New(
		llmrouter.WithProvider("openai", openai.NewFromEnv("openai", "OPENAI_API_KEY")),
		llmrouter.WithProvider("anthropic", anthropic.NewFromEnv()),
		llmrouter.WithMiddleware(
			middleware.NewRetryMiddleware(3, time.Second),
			middleware.NewTimeoutMiddleware(60*time.Second),
		),
	)

	fmt.Println("Registered providers:", router.Providers())

	// Simple completion with OpenAI
	fmt.Println("\n--- OpenAI Completion ---")
	resp, err := router.Complete(ctx, &llmrouter.Request{
		Model: "gpt-4o-mini",
		Messages: []llmrouter.Message{
			{Role: llmrouter.RoleUser, Content: "Hello! Say hi in exactly 5 words."},
		},
	})
	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}

	fmt.Printf("Response from %s:\n", resp.Provider)
	fmt.Println(resp.Choices[0].Message.Content)
	if resp.Usage != nil {
		fmt.Printf("Tokens: %d prompt, %d completion\n", resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	}

	// Simple completion with Anthropic
	fmt.Println("\n--- Anthropic Completion ---")
	resp, err = router.Complete(ctx, &llmrouter.Request{
		Model: "claude-sonnet-4-20250514",
		Messages: []llmrouter.Message{
			{Role: llmrouter.RoleUser, Content: "Hello! Say hi in exactly 5 words."},
		},
	})
	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}

	fmt.Printf("Response from %s:\n", resp.Provider)
	fmt.Println(resp.Choices[0].Message.Content)
	if resp.Usage != nil {
		fmt.Printf("Tokens: %d prompt, %d completion\n", resp.Usage.PromptTokens, resp.Usage.CompletionTokens)
	}
}
