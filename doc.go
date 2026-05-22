// Copyright 2025 bluefunda
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package llmrouter provides a unified interface for routing LLM requests
// across multiple AI providers. Write once against a single API and deploy
// across OpenAI, Anthropic Claude, Google Gemini, or any OpenAI-compatible
// service — DeepSeek, Groq, Together AI, Ollama, Sarvam, and more.
//
// # Installation
//
//	go get github.com/bluefunda/llmrouter
//
// # Quick start
//
//	import (
//	    llmrouter "github.com/bluefunda/llmrouter"
//	    "github.com/bluefunda/llmrouter/middleware"
//	    "github.com/bluefunda/llmrouter/providers/anthropic"
//	    "github.com/bluefunda/llmrouter/providers/openai"
//	)
//
//	router := llmrouter.New(
//	    llmrouter.WithProvider("openai", openai.NewFromEnv("openai", "OPENAI_API_KEY")),
//	    llmrouter.WithProvider("anthropic", anthropic.NewFromEnv()),
//	    llmrouter.WithMiddleware(
//	        middleware.Retry(3, time.Second),
//	        middleware.Timeout(60*time.Second),
//	    ),
//	)
//
//	resp, err := router.Complete(ctx, &llmrouter.Request{
//	    Model:    "gpt-4o-mini",
//	    Messages: []llmrouter.Message{{Role: llmrouter.RoleUser, Content: "Hello!"}},
//	})
//
// # Providers
//
// Three native provider packages are included:
//
//   - [github.com/bluefunda/llmrouter/providers/openai] — OpenAI (gpt-4o, gpt-4o-mini, o1, ...)
//   - [github.com/bluefunda/llmrouter/providers/anthropic] — Anthropic Claude (claude-sonnet-4, claude-haiku-4, ...)
//   - [github.com/bluefunda/llmrouter/providers/gemini] — Google Gemini (gemini-2.0-flash, gemini-2.5-pro, ...)
//
// The openai package also covers any OpenAI-compatible API via built-in presets:
//
//	openai.NewFromEnv("deepseek", "DEEPSEEK_API_KEY")   // DeepSeek
//	openai.NewFromEnv("groq",     "GROQ_API_KEY")       // Groq
//	openai.NewFromEnv("together", "TOGETHER_API_KEY")   // Together AI
//	openai.NewFromEnv("ollama",   "")                   // Ollama (local)
//	openai.NewFromEnv("sarvam",   "SARVAM_API_KEY")     // Sarvam
//
// # Streaming
//
// Use [Router.Stream] to receive tokens as they arrive:
//
//	stream, err := router.Stream(ctx, &llmrouter.Request{
//	    Model:    "claude-sonnet-4-20250514",
//	    Messages: []llmrouter.Message{{Role: llmrouter.RoleUser, Content: "Write a haiku."}},
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer stream.Close()
//	for stream.Next() {
//	    event := stream.Event()
//	    switch event.Type {
//	    case llmrouter.EventContentDelta:
//	        fmt.Print(event.Content)
//	    case llmrouter.EventDone:
//	        fmt.Println()
//	    }
//	}
//	if err := stream.Err(); err != nil {
//	    log.Fatal(err)
//	}
//
// # Fallback routing
//
// Register multiple providers and declare a fallback order. On primary failure
// the router tries each fallback in sequence, returning the first success:
//
//	router := llmrouter.New(
//	    llmrouter.WithProvider("openai",    openai.NewFromEnv("openai", "OPENAI_API_KEY")),
//	    llmrouter.WithProvider("anthropic", anthropic.NewFromEnv()),
//	    llmrouter.WithModelMapping("gpt-4o", "openai"),
//	    llmrouter.WithFallback("anthropic"), // tried if openai fails
//	)
//
// # Prompt caching
//
// Mark static blocks for provider-level caching. Anthropic uses explicit
// cache_control annotations; OpenAI and Gemini cache automatically. Observe
// savings via [Usage.CachedPromptTokens] and [Usage.CacheCreationTokens]:
//
//	req := &llmrouter.Request{
//	    Model: "claude-sonnet-4-20250514",
//	    Messages: []llmrouter.Message{
//	        {
//	            Role:         llmrouter.RoleSystem,
//	            Content:      longSystemPrompt, // paid once, reused on every call
//	            CacheControl: &llmrouter.CacheControl{Type: "ephemeral"},
//	        },
//	        {Role: llmrouter.RoleUser, Content: userQuery},
//	    },
//	}
//	resp, _ := router.Complete(ctx, req)
//	fmt.Printf("cached=%d creation=%d\n",
//	    resp.Usage.CachedPromptTokens, resp.Usage.CacheCreationTokens)
//
// # Tool calling
//
// Pass tool definitions in the request; the model returns tool calls which your
// code executes and returns as [RoleTool] messages:
//
//	req := &llmrouter.Request{
//	    Model: "gpt-4o-mini",
//	    Messages: []llmrouter.Message{
//	        {Role: llmrouter.RoleUser, Content: "What's the weather in Tokyo?"},
//	    },
//	    Tools: []llmrouter.Tool{weatherTool},
//	}
//	resp, _ := router.Complete(ctx, req)
//	if resp.Choices[0].FinishReason == "tool_calls" {
//	    tc := resp.Choices[0].Message.ToolCalls[0]
//	    result := callWeatherAPI(tc.Function.Arguments)
//	    // send result back in a follow-up request
//	}
//
// # Middleware
//
// Middleware is applied in declaration order; each wraps the next. The
// [github.com/bluefunda/llmrouter/middleware] package provides three built-ins:
//
//   - [github.com/bluefunda/llmrouter/middleware.Retry] — exponential backoff on retryable errors (429, 5xx)
//   - [github.com/bluefunda/llmrouter/middleware.Timeout] — per-request context deadline
//   - [github.com/bluefunda/llmrouter/middleware.NewCircuitBreaker] — open circuit after N consecutive failures
//
// Custom middleware is a [MiddlewareFunc] — a function that wraps a Provider:
//
//	func Logging(next llmrouter.Provider) llmrouter.Provider {
//	    return &loggingProvider{Provider: next}
//	}
//
//	router := llmrouter.New(
//	    llmrouter.WithMiddleware(Logging),
//	)
//
// # Model resolution
//
// The router resolves a model name to a provider in this order:
//
//  1. Explicit mapping via [WithModelMapping]
//  2. Provider name match (model name equals a registered provider name)
//  3. Provider model list scan via [Provider].Models()
//
// # Error handling
//
// Errors are classified for intelligent retry decisions. Use [IsRetryable] and
// [IsRateLimited] for programmatic checks, or match typed sentinels directly:
//
//	resp, err := router.Complete(ctx, req)
//	if errors.Is(err, llmrouter.ErrRateLimited) {
//	    // back off and retry later
//	}
//	if errors.Is(err, llmrouter.ErrCircuitOpen) {
//	    // provider is temporarily unavailable
//	}
//
// Other sentinels: [ErrUnknownModel], [ErrNoProviders], [ErrAuthFailed],
// [ErrMaxRetriesExceeded].
//
// # Packages
//
//   - [github.com/bluefunda/llmrouter/middleware] — retry, timeout, and circuit breaker middleware
//   - [github.com/bluefunda/llmrouter/providers/openai] — OpenAI and OpenAI-compatible providers (DeepSeek, Groq, Together AI, Ollama, Sarvam)
//   - [github.com/bluefunda/llmrouter/providers/anthropic] — Anthropic Claude
//   - [github.com/bluefunda/llmrouter/providers/gemini] — Google Gemini
package llmrouter
