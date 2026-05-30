package main

import (
	"context"
	"fmt"
	"os"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
	"github.com/bluefunda/llmrouter/middleware"
	"github.com/bluefunda/llmrouter/providers/anthropic"
)

func main() {
	ctx := context.Background()

	// Create router with Anthropic
	router := llmrouter.New(
		llmrouter.WithProvider("anthropic", anthropic.NewFromEnv()),
		llmrouter.WithMiddleware(
			middleware.Timeout(60*time.Second),
		),
	)

	fmt.Println("--- Streaming Response ---")
	fmt.Println()

	stream, err := router.Stream(ctx, &llmrouter.Request{
		Model: "claude-sonnet-4-20250514",
		Messages: []llmrouter.Message{
			{Role: llmrouter.RoleUser, Content: "Write a haiku about Go programming. Think step by step."},
		},
	})
	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}
	defer stream.Close() //nolint:errcheck

	var totalContent string
	for stream.Next() {
		event := stream.Event()
		switch event.Type {
		case llmrouter.EventContentDelta:
			fmt.Print(event.Content)
			totalContent += event.Content

		case llmrouter.EventToolCallDelta:
			fmt.Printf("\n[Tool call: %s]\n", event.Delta.ToolCalls[0].Function.Name)

		case llmrouter.EventDone:
			fmt.Println("\n\n--- Stream Complete ---")
			if event.Response != nil && event.Response.Usage != nil {
				fmt.Printf("Tokens: %d prompt, %d completion\n",
					event.Response.Usage.PromptTokens,
					event.Response.Usage.CompletionTokens)
			}
		}
	}
	if err := stream.Err(); err != nil {
		fmt.Printf("\nError: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nTotal characters streamed: %d\n", len(totalContent))
}
