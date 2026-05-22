# llmrouter

[![Go Reference](https://pkg.go.dev/badge/github.com/bluefunda/llmrouter.svg)](https://pkg.go.dev/github.com/bluefunda/llmrouter)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Go Report Card](https://goreportcard.com/badge/github.com/bluefunda/llmrouter)](https://goreportcard.com/report/github.com/bluefunda/llmrouter)

A Go library that provides a unified interface for routing requests across multiple LLM providers. Write against one API, deploy across OpenAI, Anthropic, Google Gemini, and any OpenAI-compatible service.

## Prerequisites

- Go 1.25+
- API key for at least one supported provider

## Installation

```bash
go get github.com/bluefunda/llmrouter
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "time"

    llmrouter "github.com/bluefunda/llmrouter"
    "github.com/bluefunda/llmrouter/middleware"
    "github.com/bluefunda/llmrouter/providers/anthropic"
    "github.com/bluefunda/llmrouter/providers/openai"
)

func main() {
    router := llmrouter.New(
        llmrouter.WithProvider("openai", openai.NewFromEnv("openai", "OPENAI_API_KEY")),
        llmrouter.WithProvider("anthropic", anthropic.NewFromEnv()),
        llmrouter.WithMiddleware(
            middleware.Retry(3, time.Second),
            middleware.Timeout(60*time.Second),
        ),
    )

    resp, err := router.Complete(context.Background(), &llmrouter.Request{
        Model: "gpt-4o-mini",
        Messages: []llmrouter.Message{
            {Role: llmrouter.RoleUser, Content: "Hello!"},
        },
    })
    if err != nil {
        panic(err)
    }

    fmt.Println(resp.Choices[0].Message.Content)
}
```

## Providers

Each provider is configured via environment variables or explicit options.

| Provider   | Package                                  | Env Variable         | Models                                          |
|------------|------------------------------------------|----------------------|-------------------------------------------------|
| OpenAI     | `providers/openai`                       | `OPENAI_API_KEY`     | gpt-4o, gpt-4o-mini, gpt-4.1, o4-mini          |
| Anthropic  | `providers/anthropic`                    | `ANTHROPIC_API_KEY`  | claude-opus-4, claude-sonnet-4, claude-haiku-3.5 |
| Gemini     | `providers/gemini`                       | `GEMINI_API_KEY`     | gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp |
| DeepSeek   | `providers/openai` (preset: `deepseek`)  | `DEEPSEEK_API_KEY`   | deepseek-chat, deepseek-coder                   |
| Groq       | `providers/openai` (preset: `groq`)      | `GROQ_API_KEY`       | llama-3.3-70b-versatile, mixtral-8x7b           |
| Together   | `providers/openai` (preset: `together`)  | `TOGETHER_API_KEY`   | llama-3.3-70b, mixtral-8x7b                     |
| Ollama     | `providers/openai` (preset: `ollama`)    | —                    | Any locally hosted model                        |
| Sarvam     | `providers/openai` (preset: `sarvam`)    | `SARVAM_API_KEY`     | sarvam-m, sarvam-30b, sarvam-105b               |

### OpenAI-compatible providers

DeepSeek, Groq, Together AI, Ollama, and Sarvam all use the OpenAI provider with a preset name:

```go
openai.NewFromEnv("deepseek", "DEEPSEEK_API_KEY")
openai.NewFromEnv("groq", "GROQ_API_KEY")
openai.NewFromEnv("ollama", "")  // no key needed for local
openai.NewFromEnv("sarvam", "SARVAM_API_KEY")
```

### Gemini

Gemini requires explicit error handling at construction time and holds a gRPC connection — call `router.Close()` (or `provider.Close()` directly) on shutdown:

```go
geminiProvider, err := gemini.NewFromEnv()
if err != nil {
    log.Fatal(err)
}
defer router.Close()
```

## Configuration

### Router options

```go
router := llmrouter.New(
    llmrouter.WithProvider("openai", openaiProvider),
    llmrouter.WithProvider("anthropic", anthropicProvider),
    llmrouter.WithModelMapping("gpt-4o", "openai"),
    llmrouter.WithModelMapping("claude-sonnet-4-20250514", "anthropic"),
    llmrouter.WithFallback("openai", "anthropic"),
    llmrouter.WithMiddleware(retryMw, cb.Wrap, timeoutMw),
)
```

| Option             | Description                                           |
|--------------------|-------------------------------------------------------|
| `WithProvider`     | Register a named provider                             |
| `WithModelMapping` | Route a model name to a specific provider             |
| `WithFallback`     | Set fallback provider order on primary failure        |
| `WithMiddleware`   | Attach middleware to the processing chain             |

### Model resolution

The router resolves a model to a provider in this order:

1. **Explicit mapping** — `WithModelMapping("gpt-4o", "openai")`
2. **Provider name match** — model name equals a registered provider name
3. **Provider model list** — iterates providers in registration order and checks `Models()`

## Middleware

Middleware is a `MiddlewareFunc` — a plain `func(Provider) Provider`. It is applied in declaration order (first declared = outermost wrapper).

### Retry

Exponential backoff with configurable max attempts. Non-retryable errors (auth failures, invalid requests, context cancellation) short-circuit immediately.

```go
middleware.Retry(3, time.Second)
middleware.Retry(3, time.Second, middleware.WithMaxDelay(10*time.Second))
middleware.Retry(3, time.Second, middleware.WithRetryFunc(myRetryPolicy))
```

### Circuit Breaker

Stdlib-only three-state circuit breaker (Closed → Open → HalfOpen). Opens after consecutive failures exceed the threshold; recovers after the timeout period. No external dependencies.

Because the circuit breaker has observable state, it is constructed separately and passed via `cb.Wrap`:

```go
cb := middleware.NewCircuitBreaker(5, 30*time.Second)
router := llmrouter.New(
    llmrouter.WithMiddleware(cb.Wrap),
)
fmt.Println(cb.State()) // CBStateClosed / CBStateOpen / CBStateHalfOpen
```

### Timeout

Enforces a deadline on both `Complete` and `Stream` calls. On timeout, `Stream` surfaces the error through `StreamResult.Err()`.

```go
middleware.Timeout(60 * time.Second)
```

### Custom middleware

Any `func(llmrouter.Provider) llmrouter.Provider` satisfies `MiddlewareFunc` directly:

```go
func Logging(next llmrouter.Provider) llmrouter.Provider {
    return &loggingProvider{Provider: next}
}

router := llmrouter.New(llmrouter.WithMiddleware(Logging))
```

## Streaming

`Stream` returns a `*StreamResult` iterator. Advance it with `Next()`, read the current event with `Event()`, and check errors after the loop with `Err()`. Always `defer stream.Close()` to release resources.

```go
stream, err := router.Stream(ctx, &llmrouter.Request{
    Model:    "claude-sonnet-4-20250514",
    Messages: []llmrouter.Message{
        {Role: llmrouter.RoleUser, Content: "Write a haiku about Go."},
    },
})
if err != nil {
    log.Fatal(err)
}
defer stream.Close()

for stream.Next() {
    event := stream.Event()
    switch event.Type {
    case llmrouter.EventContentDelta:
        fmt.Print(event.Content)
    case llmrouter.EventToolCallDelta:
        // handle tool call delta
    case llmrouter.EventDone:
        // event.Response holds the final response with usage stats
    }
}
if err := stream.Err(); err != nil {
    log.Fatal(err)
}
```

## Tool Calling

Define tools once and use them across any provider that supports function calling:

```go
tool := llmrouter.Tool{
    Type: "function",
    Function: llmrouter.Function{
        Name:        "get_weather",
        Description: "Get current weather for a location",
        Parameters:  json.RawMessage(`{
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }`),
    },
}

resp, _ := router.Complete(ctx, &llmrouter.Request{
    Model:    "gpt-4o-mini",
    Messages: messages,
    Tools:    []llmrouter.Tool{tool},
})
```

## Multimodal

Messages support text, images, and documents via `ContentParts`:

```go
msg := llmrouter.Message{
    Role: llmrouter.RoleUser,
    ContentParts: []llmrouter.ContentPart{
        {Type: "text", Text: "What's in this image?"},
        {Type: "image_url", ImageURL: &llmrouter.ImageURL{URL: "https://..."}},
    },
}
```

## Prompt Caching

Mark static content for provider-level caching. Anthropic uses explicit `CacheControl` annotations; OpenAI and Gemini cache automatically. Observe savings via `Usage.CachedPromptTokens`:

```go
req := &llmrouter.Request{
    Model: "claude-sonnet-4-20250514",
    Messages: []llmrouter.Message{
        {
            Role:         llmrouter.RoleSystem,
            Content:      longSystemPrompt,
            CacheControl: &llmrouter.CacheControl{Type: "ephemeral"},
        },
        {Role: llmrouter.RoleUser, Content: userQuery},
    },
}
```

## Error Handling

The library classifies errors for intelligent retry and routing decisions:

| Error                   | Retryable | Description                     |
|-------------------------|-----------|---------------------------------|
| `ErrRateLimited`        | Yes       | Provider rate limit (429)       |
| `ErrAuthFailed`         | No        | Invalid API key (401/403)       |
| `ErrInvalidRequest`     | No        | Malformed request (400)         |
| `ErrCircuitOpen`        | No        | Circuit breaker is open         |
| `ErrMaxRetriesExceeded` | No        | All retry attempts exhausted    |
| `ErrUnknownModel`       | No        | Model not found in any provider |
| `ErrNoProviders`        | No        | No providers registered         |

Use `llmrouter.IsRetryable(err)` and `llmrouter.IsRateLimited(err)` for programmatic checks.

## Project Structure

```
router.go                      # Core router — provider registry, model resolution, middleware chain
provider.go                    # Provider interface and MiddlewareFunc type
types.go                       # Unified request/response types, streaming events, tool definitions
options.go                     # Functional options for router configuration
errors.go                      # Error types and retryability classification
middleware/
  retry.go                     # Retry with exponential backoff
  timeout.go                   # Request timeout enforcement
  breaker.go                   # Circuit breaker state machine (stdlib only)
  circuitbreaker.go            # Circuit breaker middleware wrapper
providers/
  openai/                      # OpenAI + compatible providers (DeepSeek, Groq, Together, Ollama, Sarvam)
  anthropic/                   # Anthropic Claude
  gemini/                      # Google Gemini
examples/
  simple/                      # Basic completion
  streaming/                   # Streaming responses
  tools/                       # Function calling
  fallback/                    # Multi-provider with middleware
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

Built by BlueFunda — open-sourced under Apache 2.0.
