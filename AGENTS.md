# AGENTS.md

## Project Overview

Go library (`github.com/bluefunda/llmrouter`) that provides a unified interface for routing LLM requests across multiple providers (OpenAI, Anthropic, Gemini, and OpenAI-compatible services). This is a library, not a service.

## Build and Test

```bash
go build ./...
go test ./...
go test -race ./...
go vet ./...
golangci-lint run
```

All five must pass before any change is considered complete.

## Architecture

### Core interfaces (root package `llmrouter`)

| File | Purpose |
|---|---|
| `provider.go` | `Provider` and `Middleware` interfaces — all providers and middleware implement these |
| `router.go` | `Router` struct — provider registry, model resolution (3-step: explicit mapping → name match → model list scan), middleware chain construction |
| `types.go` | Unified request/response types (OpenAI-compatible), streaming events, tool definitions |
| `options.go` | Functional options: `WithProvider`, `WithModelMapping`, `WithFallback`, `WithMiddleware` |
| `errors.go` | Sentinel errors, `APIError` type, `IsRetryable()`/`IsRateLimited()` classification |

### Providers (`providers/`)

| Package | What it covers |
|---|---|
| `providers/openai/` | OpenAI and all OpenAI-compatible APIs (DeepSeek, Groq, Together, Ollama, Sarvam) via the `Presets` map |
| `providers/anthropic/` | Anthropic Claude |
| `providers/gemini/` | Google Gemini |

Each provider package contains:
- `provider.go` — implements the `Provider` interface, handles API calls
- `converter.go` — converts between unified types and provider-specific SDK types

### Middleware (`middleware/`)

| File | What it does |
|---|---|
| `retry.go` | Exponential backoff retry, respects `IsRetryable()` |
| `timeout.go` | Context deadline enforcement |
| `circuitbreaker.go` | Circuit breaker via `sony/gobreaker` |

### Other directories

- `examples/` — usage examples (simple, streaming, tools, fallback). Read-only reference; do not modify unless the public API changes.

## Modification Rules

### Adding a new OpenAI-compatible provider

1. Add a preset entry to the `Presets` map in `providers/openai/provider.go`
2. Optionally add a convenience constructor (`NewXxx`) in the same file
3. Update `README.md` provider table
4. Do NOT create a new package under `providers/` for OpenAI-compatible APIs

### Adding a new non-OpenAI provider

1. Create a new package under `providers/<name>/`
2. Implement the `Provider` interface from `provider.go`
3. Include `provider.go` and `converter.go` in the new package
4. Add tests in `provider_test.go`

### Adding new middleware

1. Add a new file in `middleware/`
2. Implement the `Middleware` interface (`Wrap(Provider) Provider`)
3. The wrapped provider must delegate `Name()` and `Models()` to the inner provider

### Modifying unified types

Files `types.go`, `provider.go`, and `errors.go` define the public API surface. Changes here affect all providers and all consumers. When modifying these files:
- Ensure all provider `converter.go` files are updated to handle new fields
- Maintain JSON tag compatibility (OpenAI-compatible format)
- Keep `Request`/`Response` structs backward-compatible (add fields, do not remove or rename)

## Code Conventions

- Module requires Go 1.24 (`go.mod`)
- All `.go` files must have the Apache 2.0 license header (`// Copyright 2025 bluefunda`). Do NOT add license headers to `.md` files.
- Use `t.Context()` in tests (available since Go 1.24), not `context.Background()`
- Errors follow the sentinel pattern with `errors.Is`/`errors.As`; wrap with `%w`
- Router is thread-safe via `sync.RWMutex`; maintain this invariant
- Provider SDKs are at alpha versions; do not upgrade without verifying API compatibility
- No generics in use; keep it that way unless there is a clear benefit
- `go.sum` is committed; run `go mod tidy` after dependency changes

## Test Patterns

- Tests use `net/http/httptest` to mock provider APIs (see `providers/openai/provider_test.go`)
- Return minimal valid JSON responses from mock servers
- Test coverage is currently minimal — add tests for any new or modified code
- Test files go in the same package as the code they test

## CI/CD

- CI runs on PRs to `main` and pushes to `main` via an inline OSS workflow (no external reusable workflow dependency)
- Releases are automated via `release-please` on the `main` branch
- Conventional commit messages are required for release-please to work correctly

## Git Conventions

- Commits: conventional format (`feat:`, `fix:`, `chore:`, `docs:`)
- Branches: `<type>/<short-description>`
- PRs: squash-merged to `main`

## Known Gaps

- The `fallbacks` field in `Router` is declared and configurable via `WithFallback` but not used during request routing — `resolveProvider` does not attempt fallback providers on failure
- Only `providers/openai/` has tests
