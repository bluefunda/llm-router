package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	llmrouter "github.com/bluefunda/llm-router"
	"github.com/bluefunda/llm-router/middleware"
	"github.com/bluefunda/llm-router/providers/openai"
)

func main() {
	ctx := context.Background()

	// Create router with OpenAI
	router := llmrouter.New(
		llmrouter.WithProvider("openai", openai.NewFromEnv("openai", "OPENAI_API_KEY")),
		llmrouter.WithMiddleware(
			middleware.NewTimeoutMiddleware(60*time.Second),
		),
	)

	// Define a tool
	weatherTool := llmrouter.Tool{
		Type: "function",
		Function: llmrouter.Function{
			Name:        "get_weather",
			Description: "Get the current weather for a location",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"location": {
						"type": "string",
						"description": "The city and state, e.g., San Francisco, CA"
					},
					"unit": {
						"type": "string",
						"enum": ["celsius", "fahrenheit"],
						"description": "The temperature unit"
					}
				},
				"required": ["location"]
			}`),
		},
	}

	fmt.Println("--- Tool Calling Example ---")
	fmt.Println()

	// First request - AI will call the tool
	resp, err := router.Complete(ctx, &llmrouter.Request{
		Model: "gpt-4o-mini",
		Messages: []llmrouter.Message{
			{Role: llmrouter.RoleUser, Content: "What's the weather like in San Francisco?"},
		},
		Tools: []llmrouter.Tool{weatherTool},
	})
	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}

	fmt.Printf("Response from %s:\n", resp.Provider)
	fmt.Printf("Finish reason: %s\n", resp.Choices[0].FinishReason)

	if len(resp.Choices[0].Message.ToolCalls) > 0 {
		tc := resp.Choices[0].Message.ToolCalls[0]
		fmt.Printf("\nTool call requested:\n")
		fmt.Printf("  Function: %s\n", tc.Function.Name)
		fmt.Printf("  Arguments: %s\n", tc.Function.Arguments)

		// Simulate tool execution
		toolResult := `{"temperature": 68, "unit": "fahrenheit", "condition": "sunny"}`
		fmt.Printf("\nSimulated tool result: %s\n", toolResult)

		// Second request - send tool result back
		resp, err = router.Complete(ctx, &llmrouter.Request{
			Model: "gpt-4o-mini",
			Messages: []llmrouter.Message{
				{Role: llmrouter.RoleUser, Content: "What's the weather like in San Francisco?"},
				{
					Role:      llmrouter.RoleAssistant,
					ToolCalls: resp.Choices[0].Message.ToolCalls,
				},
				{
					Role:       llmrouter.RoleTool,
					Content:    toolResult,
					ToolCallID: tc.ID,
				},
			},
			Tools: []llmrouter.Tool{weatherTool},
		})
		if err != nil {
			fmt.Println("Error:", err)
			os.Exit(1)
		}

		fmt.Printf("\nFinal response:\n%s\n", resp.Choices[0].Message.Content)
	} else {
		fmt.Printf("Content: %s\n", resp.Choices[0].Message.Content)
	}
}
