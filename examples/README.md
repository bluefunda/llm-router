# Examples

Runnable examples for `github.com/bluefunda/llmrouter`. Each example requires at least one provider API key set as an environment variable.

## simple

Basic completion request against OpenAI and Anthropic using retry and timeout middleware.

```bash
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... go run ./examples/simple/
```

## streaming

Streaming response using Anthropic, consuming typed events from a channel.

```bash
ANTHROPIC_API_KEY=... go run ./examples/streaming/
```

## tools

Function calling with OpenAI: first turn requests a tool call, second turn sends the tool result back.

```bash
OPENAI_API_KEY=... go run ./examples/tools/
```

## fallback

Multi-provider setup with full middleware stack (retry, circuit breaker, timeout) across OpenAI, Anthropic, DeepSeek, Groq, and optionally Gemini.

```bash
OPENAI_API_KEY=... ANTHROPIC_API_KEY=... DEEPSEEK_API_KEY=... GROQ_API_KEY=... go run ./examples/fallback/
```
