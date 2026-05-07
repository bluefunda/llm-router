# CLAUDE.md — llm-router

## What is this?

Go library (`github.com/bluefunda/llm-router`) providing a unified interface for routing LLM requests across OpenAI, Anthropic, Gemini, and OpenAI-compatible providers. This is a library, not a service.

Module: `github.com/bluefunda/llm-router` | Go 1.24

## Build & Verify

```bash
go build ./...
go test -race ./...
go vet ./...
golangci-lint run
```

All four must pass before any change is complete.

## Architecture

### Root package `llmrouter`

| File | Purpose |
|---|---|
| `router.go` | `Router` struct — provider registry, model resolution, middleware chain |
| `provider.go` | `Provider` and `Middleware` interfaces |
| `types.go` | Unified request/response types, streaming events, tool definitions |
| `options.go` | Functional options: `WithProvider`, `WithModelMapping`, `WithFallback`, `WithMiddleware` |
| `errors.go` | Sentinel errors, `APIError` type, `IsRetryable()`/`IsRateLimited()` classification |

### Providers (`providers/`)

| Package | What it covers |
|---|---|
| `providers/openai/` | OpenAI + all OpenAI-compatible APIs via the `Presets` map (DeepSeek, Groq, Together, Ollama, Sarvam) |
| `providers/anthropic/` | Anthropic Claude |
| `providers/gemini/` | Google Gemini |

### Middleware (`middleware/`)

| File | What it does |
|---|---|
| `retry.go` | Exponential backoff retry |
| `timeout.go` | Context deadline enforcement |
| `circuitbreaker.go` | Circuit breaker via `sony/gobreaker` |

## Key Rules

- **OpenAI-compatible providers** (DeepSeek, Groq, Together, Ollama, Sarvam) go in `providers/openai/` Presets map — do NOT create a new package for these
- **Public API** (`types.go`, `provider.go`, `errors.go`) is backward-compatible only: add fields, never remove or rename
- **All `.go` files** must have the Apache 2.0 header (`// Copyright 2025 bluefunda // Licensed under...`)
- **Tests**: use `t.Context()` (Go 1.24+), not `context.Background()`

## Conventions

- Commits: conventional format (`feat:`, `fix:`, `chore:`, `docs:`)
- Branches: `<type>/<short-description>`
- PRs: squash-merged to `main`

## Do NOT

- Modify CI/CD workflows without explicit request
- Break the `Provider` or `Middleware` interfaces
- Add a new `providers/<name>/` package for an OpenAI-compatible API — use `Presets` instead
- Add Apache license headers to `.md` files
