# Contributing to llmrouter

## Prerequisites

- Go 1.24+
- `golangci-lint` installed ([install guide](https://golangci-lint.run/welcome/install/))

## Getting started

```bash
git clone https://github.com/bluefunda/llmrouter.git
cd llmrouter
go build ./...
go test -race ./...
```

All four of these must pass before any PR is submitted:

```bash
go build ./...
go test -race ./...
go vet ./...
golangci-lint run
```

## Making changes

### Branch naming

`<type>/<short-description>` — e.g., `feat/sarvam-provider`, `fix/gemini-streaming`, `docs/middleware-examples`

### Adding an OpenAI-compatible provider

OpenAI-compatible APIs (DeepSeek, Groq, Together, Ollama, Sarvam, …) do **not** need a new package. Add a preset to the `Presets` map in `providers/openai/provider.go`:

```go
"myprovider": {
    BaseURL: "https://api.myprovider.com/v1",
    Models:  []string{"model-a", "model-b"},
},
```

Then add the `MYPROVIDER_API_KEY` env var note to `README.md`.

### Adding a non-OpenAI provider

1. Create `providers/<name>/provider.go` implementing the `Provider` interface from `provider.go`
2. Create `providers/<name>/converter.go` for request/response translation
3. Create `providers/<name>/doc.go` with package documentation
4. Add Apache 2.0 header to all new `.go` files
5. Add tests in `providers/<name>/provider_test.go` (use `net/http/httptest`)
6. Update `README.md` providers table

### Adding middleware

1. Create `middleware/<name>.go` implementing `Middleware` (`Wrap(Provider) Provider`)
2. The wrapped provider must delegate `Name()` and `Models()` to the inner provider
3. Add Apache 2.0 header

### Modifying public types

`types.go`, `provider.go`, and `errors.go` define the public API. Changes must be backward-compatible:

- Add fields — OK
- Remove or rename fields — not allowed without a major version bump
- All provider `converter.go` files must handle new fields

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(providers): add Mistral provider
fix(middleware): retry on 503 in addition to 429
docs: update streaming example
chore: bump golangci-lint to v2.12
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

## Pull requests

- Target `main`; PRs are squash-merged
- PR title must follow conventional commit format (enforced by CI)
- Run the full check suite locally before pushing
- Add or update tests for any changed behaviour
- All `.go` files must carry the Apache 2.0 license header

## License

By contributing you agree your work will be licensed under [Apache 2.0](LICENSE).
