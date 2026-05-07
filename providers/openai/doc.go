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
