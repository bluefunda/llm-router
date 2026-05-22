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

package anthropic

import (
	"context"
	"os"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// Provider handles Anthropic Claude API
type Provider struct {
	client *anthropic.Client
	model  string
	models []string
}

// DefaultModels is the list of available Claude models
var DefaultModels = []string{
	"claude-opus-4-20250514",
	"claude-sonnet-4-20250514",
	"claude-3-5-haiku-20241022",
	"claude-3-5-sonnet-20241022",
	"claude-3-opus-20240229",
	"claude-3-sonnet-20240229",
	"claude-3-haiku-20240307",
}

// New creates a new Anthropic provider
func New(cfg llmrouter.ProviderConfig) *Provider {
	model := cfg.Model
	if model == "" {
		model = "claude-sonnet-4-20250514"
	}

	models := cfg.Models
	if len(models) == 0 {
		models = DefaultModels
	}

	opts := []option.RequestOption{}
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}
	if cfg.Timeout > 0 {
		opts = append(opts, option.WithRequestTimeout(cfg.Timeout))
	}

	return &Provider{
		client: anthropic.NewClient(opts...),
		model:  model,
		models: models,
	}
}

// NewFromEnv creates a provider using the ANTHROPIC_API_KEY environment variable
func NewFromEnv() *Provider {
	return New(llmrouter.ProviderConfig{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
	})
}

func (p *Provider) Name() string {
	return "anthropic"
}

func (p *Provider) Models() []string {
	out := make([]string, len(p.models))
	copy(out, p.models)
	return out
}

// resolveModel returns the model name to use for a request.
func (p *Provider) resolveModel(req *llmrouter.Request) string {
	if req.Model == "" || req.Model == p.Name() {
		return p.model
	}
	return req.Model
}

// buildParams constructs the API params shared by Complete and Stream.
func (p *Provider) buildParams(req *llmrouter.Request) anthropic.MessageNewParams {
	messages, systemBlocks := convertMessages(req.Messages)

	maxTokens := int64(16384)
	if req.MaxTokens != nil {
		maxTokens = int64(*req.MaxTokens)
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.F(p.resolveModel(req)),
		MaxTokens: anthropic.F(maxTokens),
		Messages:  anthropic.F(messages),
	}

	if len(systemBlocks) > 0 {
		params.System = anthropic.F(systemBlocks)
	}
	if req.Temperature != nil {
		params.Temperature = anthropic.F(*req.Temperature)
	}
	if req.TopP != nil {
		params.TopP = anthropic.F(*req.TopP)
	}
	if len(req.Stop) > 0 {
		params.StopSequences = anthropic.F(req.Stop)
	}
	if len(req.Tools) > 0 {
		params.Tools = anthropic.F(convertTools(req.Tools))
	}
	if req.ToolChoice != nil {
		params.ToolChoice = anthropic.F(convertToolChoice(req.ToolChoice))
	}

	return params
}

func (p *Provider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	params := p.buildParams(req)

	resp, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return nil, wrapError(err)
	}

	return convertToOpenAIResponse(resp, p.Name()), nil
}

func (p *Provider) Stream(ctx context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
	model := p.resolveModel(req)
	params := p.buildParams(req)

	ctx, cancel := context.WithCancel(ctx)
	ch := make(chan llmrouter.Event)
	res := llmrouter.NewStreamResult(ch)
	res.OnClose(func() error { cancel(); return nil })

	go func() {
		defer close(ch)
		defer cancel()

		stream := p.client.Messages.NewStreaming(ctx, params)

		var fullContent string
		var toolCalls []llmrouter.ToolCall
		var currentToolID string
		var currentToolName string
		var toolArgsBuilder string
		var inputTokens, outputTokens int64
		var cacheCreationTokens, cacheReadTokens int64
		var msgID string
		var stopReason string

		for stream.Next() {
			event := stream.Current()

			switch e := event.AsUnion().(type) {
			case anthropic.MessageStartEvent:
				if e.Message.ID != "" {
					msgID = e.Message.ID
				}
				if e.Message.Usage.InputTokens > 0 {
					inputTokens = e.Message.Usage.InputTokens
				}
				cacheCreationTokens = e.Message.Usage.CacheCreationInputTokens
				cacheReadTokens = e.Message.Usage.CacheReadInputTokens

			case anthropic.ContentBlockStartEvent:
				switch cb := e.ContentBlock.AsUnion().(type) {
				case anthropic.TextBlock:
					_ = cb
				case anthropic.ToolUseBlock:
					currentToolID = cb.ID
					currentToolName = cb.Name
					toolArgsBuilder = ""
				}

			case anthropic.ContentBlockDeltaEvent:
				switch d := e.Delta.AsUnion().(type) {
				case anthropic.TextDelta:
					fullContent += d.Text
					ch <- llmrouter.Event{
						Type:    llmrouter.EventContentDelta,
						Content: d.Text,
					}
				case anthropic.InputJSONDelta:
					toolArgsBuilder += d.PartialJSON
					ch <- llmrouter.Event{
						Type: llmrouter.EventToolCallDelta,
						Delta: &llmrouter.Delta{
							ToolCalls: []llmrouter.ToolCall{
								{
									ID:   currentToolID,
									Type: "function",
									Function: llmrouter.FuncCall{
										Name:      currentToolName,
										Arguments: d.PartialJSON,
									},
								},
							},
						},
					}
				}

			case anthropic.ContentBlockStopEvent:
				if currentToolID != "" && currentToolName != "" {
					toolCalls = append(toolCalls, llmrouter.ToolCall{
						ID:   currentToolID,
						Type: "function",
						Function: llmrouter.FuncCall{
							Name:      currentToolName,
							Arguments: toolArgsBuilder,
						},
					})
					currentToolID = ""
					currentToolName = ""
					toolArgsBuilder = ""
				}

			case anthropic.MessageDeltaEvent:
				if e.Delta.StopReason != "" {
					stopReason = string(e.Delta.StopReason)
				}
				if e.Usage.OutputTokens > 0 {
					outputTokens = e.Usage.OutputTokens
				}
			}
		}

		if err := stream.Err(); err != nil {
			ch <- llmrouter.Event{
				Type:  llmrouter.EventError,
				Error: wrapError(err),
			}
			return
		}

		finishReason := "stop"
		switch stopReason {
		case "tool_use":
			finishReason = "tool_calls"
		case "max_tokens":
			finishReason = "length"
		}

		ch <- llmrouter.Event{
			Type: llmrouter.EventDone,
			Response: &llmrouter.Response{
				ID:       msgID,
				Object:   "chat.completion",
				Model:    model,
				Provider: p.Name(),
				Created:  time.Now().Unix(),
				Choices: []llmrouter.Choice{
					{
						Index: 0,
						Message: &llmrouter.Message{
							Role:      llmrouter.RoleAssistant,
							Content:   fullContent,
							ToolCalls: toolCalls,
						},
						FinishReason: finishReason,
					},
				},
				Usage: &llmrouter.Usage{
					PromptTokens:        int(inputTokens),
					CompletionTokens:    int(outputTokens),
					TotalTokens:         int(inputTokens + outputTokens),
					CachedPromptTokens:  int(cacheReadTokens),
					CacheCreationTokens: int(cacheCreationTokens),
				},
			},
		}
	}()

	return res, nil
}
