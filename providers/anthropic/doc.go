// Package anthropic implements the llmrouter.Provider interface for
// Anthropic Claude models using the official Anthropic Go SDK.
//
// # Usage
//
//	p := anthropic.NewFromEnv() // reads ANTHROPIC_API_KEY
//
// Or with an explicit key:
//
//	p := anthropic.New(anthropic.Config{APIKey: "sk-ant-..."})
//
// Supported models include claude-opus-4, claude-sonnet-4,
// claude-haiku-4, and claude-haiku-3.5. The provider supports
// streaming, tool calling, and multimodal inputs (text + images).
package anthropic
