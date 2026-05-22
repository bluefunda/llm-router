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

package llmrouter

import (
	"context"
)

// Provider is the core interface that all LLM providers must implement.
type Provider interface {
	// Name returns the provider identifier (e.g., "openai", "anthropic")
	Name() string

	// Models returns the list of supported model IDs
	Models() []string

	// Complete performs a non-streaming completion
	Complete(ctx context.Context, req *Request) (*Response, error)

	// Stream performs a streaming completion
	Stream(ctx context.Context, req *Request) (*StreamResult, error)
}

// MiddlewareFunc wraps a Provider with additional functionality.
// It is a plain function type; any func(Provider) Provider satisfies it directly.
type MiddlewareFunc func(Provider) Provider
