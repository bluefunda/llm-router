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
// service (DeepSeek, Groq, Together AI, Ollama, Sarvam).
//
// # Installation
//
//	go get github.com/bluefunda/llm-router
//
// # Quick start
//
//	router := llmrouter.New(
//	    llmrouter.WithProvider("openai", openai.NewFromEnv("openai", "OPENAI_API_KEY")),
//	    llmrouter.WithProvider("anthropic", anthropic.NewFromEnv()),
//	    llmrouter.WithMiddleware(
//	        middleware.NewRetryMiddleware(3, time.Second),
//	        middleware.NewTimeoutMiddleware(60*time.Second),
//	    ),
//	)
//
//	resp, err := router.Complete(ctx, &llmrouter.Request{
//	    Model:    "gpt-4o-mini",
//	    Messages: []llmrouter.Message{{Role: llmrouter.RoleUser, Content: "Hello!"}},
//	})
//
// # Packages
//
// The root package contains the router, provider interface, and unified types.
// Sub-packages provide concrete implementations:
//
//   - [github.com/bluefunda/llm-router/middleware] — retry, timeout, and circuit breaker middleware
//   - [github.com/bluefunda/llm-router/providers/openai] — OpenAI and OpenAI-compatible providers
//     (DeepSeek, Groq, Together AI, Ollama, Sarvam)
//   - [github.com/bluefunda/llm-router/providers/anthropic] — Anthropic Claude
//   - [github.com/bluefunda/llm-router/providers/gemini] — Google Gemini
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
// [IsRateLimited] for programmatic checks, or inspect typed errors such as
// [ErrRateLimited], [ErrAuthFailed], and [ErrCircuitOpen].
package llmrouter
