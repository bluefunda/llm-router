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

package gemini

import (
	"context"
	"os"
	"time"

	llmrouter "github.com/bluefunda/llmrouter"
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

// Provider handles Google Gemini API
type Provider struct {
	client *genai.Client
	model  string
	models []string
}

// DefaultModels is the list of available Gemini models
var DefaultModels = []string{
	"gemini-1.5-pro",
	"gemini-1.5-flash",
	"gemini-2.0-flash-exp",
	"gemini-1.0-pro",
}

// New creates a new Gemini provider.
// The Gemini SDK requires a context for client construction; context.Background()
// is used internally so the provider lifetime is not tied to a caller's context.
func New(cfg llmrouter.ProviderConfig) (*Provider, error) {
	model := cfg.Model
	if model == "" {
		model = "gemini-1.5-flash"
	}

	models := cfg.Models
	if len(models) == 0 {
		models = DefaultModels
	}

	opts := []option.ClientOption{}
	if cfg.APIKey != "" {
		opts = append(opts, option.WithAPIKey(cfg.APIKey))
	}

	client, err := genai.NewClient(context.Background(), opts...)
	if err != nil {
		return nil, err
	}

	return &Provider{
		client: client,
		model:  model,
		models: models,
	}, nil
}

// NewFromEnv creates a provider using the GEMINI_API_KEY environment variable
func NewFromEnv() (*Provider, error) {
	return New(llmrouter.ProviderConfig{
		APIKey: os.Getenv("GEMINI_API_KEY"),
	})
}

// Close closes the Gemini client
func (p *Provider) Close() error {
	return p.client.Close()
}

func (p *Provider) Name() string {
	return "gemini"
}

func (p *Provider) Models() []string {
	return p.models
}

func (p *Provider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	modelName := req.Model
	if modelName == "" {
		modelName = p.model
	}

	model := p.client.GenerativeModel(modelName)
	configureModel(model, req)

	if len(req.Tools) > 0 {
		model.Tools = convertTools(req.Tools)
	}

	chat := model.StartChat()
	history, lastParts := convertHistory(req.Messages)
	chat.History = history

	resp, err := chat.SendMessage(ctx, lastParts...)
	if err != nil {
		return nil, wrapError(err)
	}

	return convertResponse(resp, modelName, p.Name()), nil
}

func (p *Provider) Stream(ctx context.Context, req *llmrouter.Request) (*llmrouter.StreamResult, error) {
	modelName := req.Model
	if modelName == "" {
		modelName = p.model
	}

	model := p.client.GenerativeModel(modelName)
	configureModel(model, req)

	if len(req.Tools) > 0 {
		model.Tools = convertTools(req.Tools)
	}

	chat := model.StartChat()
	history, lastParts := convertHistory(req.Messages)
	chat.History = history

	ctx, cancel := context.WithCancel(ctx)
	ch := make(chan llmrouter.Event)
	res := llmrouter.NewStreamResult(ch)
	res.OnClose(func() error { cancel(); return nil })

	go func() {
		defer close(ch)
		defer cancel()

		iter := chat.SendMessageStream(ctx, lastParts...)

		var fullContent string
		var toolCalls []llmrouter.ToolCall

		for {
			resp, err := iter.Next()
			if err == iterator.Done {
				break
			}
			if err != nil {
				ch <- llmrouter.Event{
					Type:  llmrouter.EventError,
					Error: wrapError(err),
				}
				return
			}

			for _, candidate := range resp.Candidates {
				if candidate.Content == nil {
					continue
				}
				for _, part := range candidate.Content.Parts {
					switch p := part.(type) {
					case genai.Text:
						content := string(p)
						fullContent += content
						ch <- llmrouter.Event{
							Type:    llmrouter.EventContentDelta,
							Content: content,
						}
					case genai.FunctionCall:
						args, _ := convertFunctionCallArgs(p.Args)
						tc := llmrouter.ToolCall{
							ID:   p.Name,
							Type: "function",
							Function: llmrouter.FuncCall{
								Name:      p.Name,
								Arguments: args,
							},
						}
						toolCalls = append(toolCalls, tc)
						ch <- llmrouter.Event{
							Type: llmrouter.EventToolCallDelta,
							Delta: &llmrouter.Delta{
								ToolCalls: []llmrouter.ToolCall{tc},
							},
						}
					}
				}
			}
		}

		finishReason := "stop"
		if len(toolCalls) > 0 {
			finishReason = "tool_calls"
		}

		ch <- llmrouter.Event{
			Type: llmrouter.EventDone,
			Response: &llmrouter.Response{
				Model:    modelName,
				Provider: p.Name(),
				Object:   "chat.completion",
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
			},
		}
	}()

	return res, nil
}

func configureModel(model *genai.GenerativeModel, req *llmrouter.Request) {
	if req.Temperature != nil {
		temp := float32(*req.Temperature)
		model.Temperature = &temp
	}
	if req.MaxTokens != nil {
		tokens := int32(*req.MaxTokens)
		model.MaxOutputTokens = &tokens
	} else {
		tokens := int32(16384)
		model.MaxOutputTokens = &tokens
	}
	if req.TopP != nil {
		topP := float32(*req.TopP)
		model.TopP = &topP
	}
	if len(req.Stop) > 0 {
		model.StopSequences = req.Stop
	}

	for _, msg := range req.Messages {
		if msg.Role == llmrouter.RoleSystem {
			model.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(msg.Content)},
			}
			break
		}
	}
}
