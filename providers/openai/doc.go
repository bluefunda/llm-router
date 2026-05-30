// Package openai implements the llmrouter.Provider interface for OpenAI and
// any OpenAI-compatible API. A single provider type covers OpenAI, DeepSeek,
// Groq, Together AI, Ollama, and Sarvam via named presets.
//
// # OpenAI
//
//	p := openai.NewFromEnv("openai", "OPENAI_API_KEY")
//
// # OpenAI-compatible providers (presets)
//
//	deepseek := openai.NewFromEnv("deepseek", "DEEPSEEK_API_KEY")
//	groq     := openai.NewFromEnv("groq",     "GROQ_API_KEY")
//	together := openai.NewFromEnv("together", "TOGETHER_API_KEY")
//	ollama   := openai.NewFromEnv("ollama",   "") // no key for local
//	sarvam   := openai.NewFromEnv("sarvam",   "SARVAM_API_KEY")
//
// Each preset configures the correct base URL, default model list, and any
// provider-specific behaviour (e.g. StringContentOnly for APIs that do not
// accept structured content arrays).
//
// # Custom endpoints
//
//	p := openai.New("my-provider", openai.Config{
//	    APIKey:  os.Getenv("MY_KEY"),
//	    BaseURL: "https://my-openai-compat.example.com/v1",
//	})
package openai
