package gemini

import (
	"context"
	"os"
	"time"

	llmrouter "github.com/bluefunda/llm-router"
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

// New creates a new Gemini provider
func New(ctx context.Context, cfg llmrouter.ProviderConfig) (*Provider, error) {
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

	client, err := genai.NewClient(ctx, opts...)
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
func NewFromEnv(ctx context.Context) (*Provider, error) {
	return New(ctx, llmrouter.ProviderConfig{
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

func (p *Provider) SupportsTools() bool {
	return true
}

func (p *Provider) Complete(ctx context.Context, req *llmrouter.Request) (*llmrouter.Response, error) {
	modelName := req.Model
	if modelName == "" {
		modelName = p.model
	}

	model := p.client.GenerativeModel(modelName)
	configureModel(model, req)

	// Convert tools if present
	if len(req.Tools) > 0 {
		model.Tools = convertTools(req.Tools)
	}

	// Build chat and get history
	chat := model.StartChat()
	history, lastMsg := convertHistory(req.Messages)
	chat.History = history

	// Generate response
	resp, err := chat.SendMessage(ctx, genai.Text(lastMsg))
	if err != nil {
		return nil, wrapError(err)
	}

	return convertResponse(resp, modelName, p.Name()), nil
}

func (p *Provider) Stream(ctx context.Context, req *llmrouter.Request) (<-chan llmrouter.Event, error) {
	ch := make(chan llmrouter.Event)

	modelName := req.Model
	if modelName == "" {
		modelName = p.model
	}

	model := p.client.GenerativeModel(modelName)
	configureModel(model, req)

	// Convert tools if present
	if len(req.Tools) > 0 {
		model.Tools = convertTools(req.Tools)
	}

	// Build chat and get history
	chat := model.StartChat()
	history, lastMsg := convertHistory(req.Messages)
	chat.History = history

	go func() {
		defer close(ch)

		iter := chat.SendMessageStream(ctx, genai.Text(lastMsg))

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
							ID:   p.Name, // Gemini doesn't have IDs, use name
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

		// Send done event with full response
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

	return ch, nil
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

	// Extract system prompt from messages
	for _, msg := range req.Messages {
		if msg.Role == llmrouter.RoleSystem {
			model.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(msg.Content)},
			}
			break
		}
	}
}
