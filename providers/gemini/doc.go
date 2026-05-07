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
