package anthropic

import (
	"context"
	"encoding/json"
	"os"
	"time"

	llmrouter "github.com/bluefunda/llm-router"
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
	return p.models
}

func (p *Provider) SupportsTools() bool {
	return true
}

func (p *Provider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	messages, systemPrompt := convertMessages(req.Messages)

	model := req.Model
	if model == "" || model == "anthropic" {
		// Use default model if not specified or if model matches provider name
		model = p.model
	}

	maxTokens := int64(16384)
	if req.MaxTokens != nil {
		maxTokens = int64(*req.MaxTokens)
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.F(model),
		MaxTokens: anthropic.F(maxTokens),
		Messages:  anthropic.F(messages),
	}

	if systemPrompt != "" {
		params.System = anthropic.F([]anthropic.TextBlockParam{
			{Type: anthropic.F(anthropic.TextBlockParamTypeText), Text: anthropic.F(systemPrompt)},
		})
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

	resp, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return nil, wrapError(err)
	}

	return convertToOpenAIResponse(resp, p.Name()), nil
}

func (p *Provider) Stream(ctx context.Context, req *llmrouter.Request) (<-chan llmrouter.Event, error) {
	ch := make(chan llmrouter.Event)

	messages, systemPrompt := convertMessages(req.Messages)

	model := req.Model
	if model == "" || model == "anthropic" {
		// Use default model if not specified or if model matches provider name
		model = p.model
	}

	maxTokens := int64(16384)
	if req.MaxTokens != nil {
		maxTokens = int64(*req.MaxTokens)
	}

	params := anthropic.MessageNewParams{
		Model:     anthropic.F(model),
		MaxTokens: anthropic.F(maxTokens),
		Messages:  anthropic.F(messages),
	}

	if systemPrompt != "" {
		params.System = anthropic.F([]anthropic.TextBlockParam{
			{Type: anthropic.F(anthropic.TextBlockParamTypeText), Text: anthropic.F(systemPrompt)},
		})
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

	go func() {
		defer close(ch)

		stream := p.client.Messages.NewStreaming(ctx, params)

		// Accumulate the response manually
		var fullContent string
		var toolCalls []llmrouter.ToolCall
		var currentToolID string
		var currentToolName string
		var toolArgsBuilder string
		var inputTokens, outputTokens int64
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

			case anthropic.ContentBlockStartEvent:
				switch cb := e.ContentBlock.AsUnion().(type) {
				case anthropic.TextBlock:
					// Text block started
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
				// If we were building a tool call, finalize it
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

		// Build final response
		finishReason := "stop"
		if stopReason == "tool_use" {
			finishReason = "tool_calls"
		} else if stopReason == "max_tokens" {
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
					PromptTokens:     int(inputTokens),
					CompletionTokens: int(outputTokens),
					TotalTokens:      int(inputTokens + outputTokens),
				},
			},
		}
	}()

	return ch, nil
}

// Helper to marshal tool args
func marshalToolArgs(args interface{}) string {
	if args == nil {
		return "{}"
	}
	b, _ := json.Marshal(args)
	return string(b)
}
