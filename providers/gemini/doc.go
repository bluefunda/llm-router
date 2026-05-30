// Package gemini implements the llmrouter.Provider interface for Google
// Gemini models using the official Google Generative AI Go SDK.
//
// Gemini requires a context during initialisation:
//
//	p, err := gemini.NewFromEnv(ctx) // reads GEMINI_API_KEY
//
// Or with an explicit key:
//
//	p, err := gemini.New(ctx, gemini.Config{APIKey: "..."})
//
// Supported models include gemini-2.0-flash, gemini-1.5-pro, and
// gemini-1.5-flash. The provider supports streaming, tool calling,
// and multimodal inputs (text + images + documents).
package gemini
