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

package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// Presets contains default configurations for OpenAI-compatible providers
var Presets = map[string]struct {
	BaseURL      string
	DefaultModel string
	Models       []string
}{
	"openai": {
		BaseURL:      "https://api.openai.com/v1/",
		DefaultModel: "gpt-4.1-mini",
		Models:       []string{"gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini", "o4-mini"},
	},
	"deepseek": {
		BaseURL:      "https://api.deepseek.com/",
		DefaultModel: "deepseek-chat",
		Models:       []string{"deepseek-chat", "deepseek-coder"},
	},
	"groq": {
		BaseURL:      "https://api.groq.com/openai/v1/",
		DefaultModel: "llama-3.3-70b-versatile",
		Models:       []string{"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"},
	},
	"together": {
		BaseURL:      "https://api.together.xyz/v1/",
		DefaultModel: "meta-llama/Llama-3.3-70B-Instruct-Turbo",
		Models:       []string{"meta-llama/Llama-3.3-70B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1"},
	},
	"ollama": {
		BaseURL:      "http://localhost:11434/v1/",
		DefaultModel: "llama3.2",
		Models:       []string{}, // Dynamic based on what's installed
	},
	"sarvam": {
		BaseURL:      "https://api.sarvam.ai/v1/",
		DefaultModel: "sarvam-30b",
		Models:       []string{"sarvam-m", "sarvam-30b", "sarvam-105b"},
	},
}

// Provider handles OpenAI and OpenAI-compatible APIs
type Provider struct {
	client *openai.Client
	name   string
	model  string
	models []string
}

// New creates a new OpenAI-compatible provider
func New(cfg llmrouter.ProviderConfig) *Provider {
	preset, hasPreset := Presets[cfg.Name]

	baseURL := cfg.BaseURL
	if baseURL == "" && hasPreset {
		baseURL = preset.BaseURL
	}
	// Ensure trailing slash so url.Parse resolves paths correctly
	if baseURL != "" && !strings.HasSuffix(baseURL, "/") {
		baseURL += "/"
	}

	model := cfg.Model
	if model == "" && hasPreset {
		model = preset.DefaultModel
	}

	opts := []option.RequestOption{}
	if baseURL != "" {
		opts = append(opts, option.WithBaseURL(baseURL))
	}
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.Timeout > 0 {
		opts = append(opts, option.WithRequestTimeout(cfg.Timeout))
	}
	for key, value := range cfg.CustomHeaders {
		opts = append(opts, option.WithHeader(key, value))
	}
	if cfg.StringContentOnly {
		opts = append(opts, option.WithMiddleware(flattenContentMiddleware))
	}

	models := cfg.Models
	if len(models) == 0 && hasPreset {
		models = preset.Models
	}

	return &Provider{
		client: openai.NewClient(opts...),
		name:   cfg.Name,
		model:  model,
		models: models,
	}
}

// NewFromEnv creates a provider using environment variable for API key
func NewFromEnv(name string, envKey string) *Provider {
	return New(llmrouter.ProviderConfig{
		Name:   name,
		APIKey: os.Getenv(envKey),
	})
}

// NewOpenAI creates a standard OpenAI provider
func NewOpenAI(apiKey string) *Provider {
	return New(llmrouter.ProviderConfig{
		Name:   "openai",
		APIKey: apiKey,
	})
}

// NewDeepSeek creates a DeepSeek provider
func NewDeepSeek(apiKey string) *Provider {
	return New(llmrouter.ProviderConfig{
		Name:   "deepseek",
		APIKey: apiKey,
	})
}

// NewGroq creates a Groq provider
func NewGroq(apiKey string) *Provider {
	return New(llmrouter.ProviderConfig{
		Name:   "groq",
		APIKey: apiKey,
	})
}

// NewTogether creates a Together AI provider
func NewTogether(apiKey string) *Provider {
	return New(llmrouter.ProviderConfig{
		Name:   "together",
		APIKey: apiKey,
	})
}

// NewOllama creates an Ollama provider
func NewOllama(baseURL string) *Provider {
	if baseURL == "" {
		baseURL = "http://localhost:11434/v1"
	}
	return New(llmrouter.ProviderConfig{
		Name:    "ollama",
		BaseURL: baseURL,
		APIKey:  "ollama", // Ollama doesn't require a real key but needs something
	})
}

func (p *Provider) Name() string {
	return p.name
}

func (p *Provider) Models() []string {
	return p.models
}

func (p *Provider) SupportsTools() bool {
	return true
}

func (p *Provider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	model := req.Model
	if model == "" || model == p.name {
		// Use default model if not specified or if model matches provider name
		model = p.model
	}

	params := openai.ChatCompletionNewParams{
		Model:    openai.F(model),
		Messages: openai.F(convertMessages(req.Messages)),
	}

	if req.Temperature != nil {
		params.Temperature = openai.F(*req.Temperature)
	}
	if req.MaxTokens != nil {
		params.MaxCompletionTokens = openai.F(int64(*req.MaxTokens))
	}
	if req.TopP != nil {
		params.TopP = openai.F(*req.TopP)
	}
	if len(req.Stop) > 0 {
		params.Stop = openai.F[openai.ChatCompletionNewParamsStopUnion](openai.ChatCompletionNewParamsStopArray(req.Stop))
	}
	if len(req.Tools) > 0 {
		params.Tools = openai.F(convertTools(req.Tools))
	}
	if req.ToolChoice != nil {
		params.ToolChoice = openai.F(convertToolChoice(req.ToolChoice))
	}

	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, wrapError(p.name, err)
	}

	return convertResponse(resp, p.name), nil
}

func (p *Provider) Stream(ctx context.Context, req *llmrouter.Request) (<-chan llmrouter.Event, error) {
	ch := make(chan llmrouter.Event)

	model := req.Model
	if model == "" || model == p.name {
		// Use default model if not specified or if model matches provider name
		model = p.model
	}

	params := openai.ChatCompletionNewParams{
		Model:    openai.F(model),
		Messages: openai.F(convertMessages(req.Messages)),
	}

	if req.Temperature != nil {
		params.Temperature = openai.F(*req.Temperature)
	}
	if req.MaxTokens != nil {
		params.MaxCompletionTokens = openai.F(int64(*req.MaxTokens))
	}
	if req.TopP != nil {
		params.TopP = openai.F(*req.TopP)
	}
	if len(req.Stop) > 0 {
		params.Stop = openai.F[openai.ChatCompletionNewParamsStopUnion](openai.ChatCompletionNewParamsStopArray(req.Stop))
	}
	if len(req.Tools) > 0 {
		params.Tools = openai.F(convertTools(req.Tools))
	}
	if req.ToolChoice != nil {
		params.ToolChoice = openai.F(convertToolChoice(req.ToolChoice))
	}

	go func() {
		defer close(ch)

		stream := p.client.Chat.Completions.NewStreaming(ctx, params)

		var lastChunk *openai.ChatCompletionChunk
		for stream.Next() {
			chunk := stream.Current()
			lastChunk = &chunk

			if len(chunk.Choices) > 0 {
				delta := chunk.Choices[0].Delta

				if delta.Content != "" {
					ch <- llmrouter.Event{
						Type:    llmrouter.EventContentDelta,
						Content: delta.Content,
					}
				}

				if len(delta.ToolCalls) > 0 {
					ch <- llmrouter.Event{
						Type: llmrouter.EventToolCallDelta,
						Delta: &llmrouter.Delta{
							ToolCalls: convertStreamToolCalls(delta.ToolCalls),
						},
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- llmrouter.Event{
				Type:  llmrouter.EventError,
				Error: wrapError(p.name, err),
			}
			return
		}

		// Send final response
		if lastChunk != nil {
			ch <- llmrouter.Event{
				Type:     llmrouter.EventDone,
				Response: convertChunkResponse(lastChunk, p.name),
			}
		} else {
			ch <- llmrouter.Event{
				Type: llmrouter.EventDone,
				Response: &llmrouter.Response{
					Provider: p.name,
					Model:    model,
					Object:   "chat.completion",
					Created:  time.Now().Unix(),
				},
			}
		}
	}()

	return ch, nil
}

// flattenContentMiddleware rewrites request bodies so that message content
// is sent as a plain string instead of an array of content parts.
// Some OpenAI-compatible APIs (e.g. Sarvam) require this format.
func flattenContentMiddleware(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	if req.Body == nil {
		return next(req)
	}

	body, err := io.ReadAll(req.Body)
	_ = req.Body.Close()
	if err != nil {
		return next(req)
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(body, &payload); err != nil {
		req.Body = io.NopCloser(bytes.NewReader(body))
		return next(req)
	}

	msgs, ok := payload["messages"].([]interface{})
	if !ok {
		req.Body = io.NopCloser(bytes.NewReader(body))
		return next(req)
	}

	for _, m := range msgs {
		msg, ok := m.(map[string]interface{})
		if !ok {
			continue
		}
		parts, ok := msg["content"].([]interface{})
		if !ok || len(parts) == 0 {
			continue
		}
		// If content is a single text part, flatten to plain string
		if len(parts) == 1 {
			if part, ok := parts[0].(map[string]interface{}); ok {
				if part["type"] == "text" {
					msg["content"] = part["text"]
				}
			}
		}
	}

	modified, err := json.Marshal(payload)
	if err != nil {
		req.Body = io.NopCloser(bytes.NewReader(body))
		return next(req)
	}

	req.Body = io.NopCloser(bytes.NewReader(modified))
	req.ContentLength = int64(len(modified))
	return next(req)
}
