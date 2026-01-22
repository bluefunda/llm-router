package openai

import (
	"encoding/json"
	"net/http"
	"time"

	llmrouter "github.com/bluefunda/llm-router"
	"github.com/openai/openai-go"
)

func convertMessages(msgs []llmrouter.Message) []openai.ChatCompletionMessageParamUnion {
	result := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs))

	for _, msg := range msgs {
		switch msg.Role {
		case llmrouter.RoleSystem:
			result = append(result, openai.SystemMessage(msg.Content))

		case llmrouter.RoleUser:
			result = append(result, openai.UserMessage(msg.Content))

		case llmrouter.RoleAssistant:
			if len(msg.ToolCalls) > 0 {
				toolCalls := make([]openai.ChatCompletionMessageToolCallParam, len(msg.ToolCalls))
				for i, tc := range msg.ToolCalls {
					toolCalls[i] = openai.ChatCompletionMessageToolCallParam{
						ID:   openai.F(tc.ID),
						Type: openai.F(openai.ChatCompletionMessageToolCallTypeFunction),
						Function: openai.F(openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      openai.F(tc.Function.Name),
							Arguments: openai.F(tc.Function.Arguments),
						}),
					}
				}
				result = append(result, openai.ChatCompletionAssistantMessageParam{
					Role:      openai.F(openai.ChatCompletionAssistantMessageParamRoleAssistant),
					Content:   openai.F([]openai.ChatCompletionAssistantMessageParamContentUnion{openai.TextPart(msg.Content)}),
					ToolCalls: openai.F(toolCalls),
				})
			} else {
				result = append(result, openai.AssistantMessage(msg.Content))
			}

		case llmrouter.RoleTool:
			result = append(result, openai.ToolMessage(msg.ToolCallID, msg.Content))
		}
	}

	return result
}

func convertTools(tools []llmrouter.Tool) []openai.ChatCompletionToolParam {
	result := make([]openai.ChatCompletionToolParam, len(tools))

	for i, tool := range tools {
		var params map[string]interface{}
		if tool.Function.Parameters != nil {
			_ = json.Unmarshal(tool.Function.Parameters, &params)
		}

		result[i] = openai.ChatCompletionToolParam{
			Type: openai.F(openai.ChatCompletionToolTypeFunction),
			Function: openai.F(openai.FunctionDefinitionParam{
				Name:        openai.F(tool.Function.Name),
				Description: openai.F(tool.Function.Description),
				Parameters:  openai.F(openai.FunctionParameters(params)),
			}),
		}
	}

	return result
}

func convertToolChoice(tc *llmrouter.ToolChoice) openai.ChatCompletionToolChoiceOptionUnionParam {
	if tc == nil {
		return nil
	}

	switch tc.Type {
	case "auto":
		return openai.ChatCompletionToolChoiceOptionBehavior(openai.ChatCompletionToolChoiceOptionBehaviorAuto)
	case "none":
		return openai.ChatCompletionToolChoiceOptionBehavior(openai.ChatCompletionToolChoiceOptionBehaviorNone)
	case "required":
		return openai.ChatCompletionToolChoiceOptionBehavior(openai.ChatCompletionToolChoiceOptionBehaviorRequired)
	case "function":
		if tc.Function != nil {
			return openai.ChatCompletionNamedToolChoiceParam{
				Type: openai.F(openai.ChatCompletionNamedToolChoiceTypeFunction),
				Function: openai.F(openai.ChatCompletionNamedToolChoiceFunctionParam{
					Name: openai.F(tc.Function.Name),
				}),
			}
		}
	}

	return nil
}

func convertResponse(resp *openai.ChatCompletion, provider string) *llmrouter.Response {
	choices := make([]llmrouter.Choice, len(resp.Choices))

	for i, choice := range resp.Choices {
		var toolCalls []llmrouter.ToolCall
		if len(choice.Message.ToolCalls) > 0 {
			toolCalls = make([]llmrouter.ToolCall, len(choice.Message.ToolCalls))
			for j, tc := range choice.Message.ToolCalls {
				toolCalls[j] = llmrouter.ToolCall{
					ID:   tc.ID,
					Type: "function",
					Function: llmrouter.FuncCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		choices[i] = llmrouter.Choice{
			Index: int(choice.Index),
			Message: &llmrouter.Message{
				Role:      llmrouter.RoleAssistant,
				Content:   choice.Message.Content,
				ToolCalls: toolCalls,
			},
			FinishReason: string(choice.FinishReason),
		}
	}

	var usage *llmrouter.Usage
	if resp.Usage.TotalTokens > 0 {
		usage = &llmrouter.Usage{
			PromptTokens:     int(resp.Usage.PromptTokens),
			CompletionTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:      int(resp.Usage.TotalTokens),
		}
	}

	return &llmrouter.Response{
		ID:       resp.ID,
		Object:   string(resp.Object),
		Created:  resp.Created,
		Model:    resp.Model,
		Choices:  choices,
		Usage:    usage,
		Provider: provider,
	}
}

func convertChunkResponse(chunk *openai.ChatCompletionChunk, provider string) *llmrouter.Response {
	choices := make([]llmrouter.Choice, len(chunk.Choices))

	for i, choice := range chunk.Choices {
		var toolCalls []llmrouter.ToolCall
		if len(choice.Delta.ToolCalls) > 0 {
			toolCalls = make([]llmrouter.ToolCall, len(choice.Delta.ToolCalls))
			for j, tc := range choice.Delta.ToolCalls {
				idx := int(tc.Index)
				toolCalls[j] = llmrouter.ToolCall{
					ID:    tc.ID,
					Type:  "function",
					Index: &idx,
					Function: llmrouter.FuncCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		choices[i] = llmrouter.Choice{
			Index: int(choice.Index),
			Delta: &llmrouter.Delta{
				Role:      llmrouter.Role(choice.Delta.Role),
				Content:   choice.Delta.Content,
				ToolCalls: toolCalls,
			},
			FinishReason: string(choice.FinishReason),
		}
	}

	var usage *llmrouter.Usage
	if chunk.Usage.TotalTokens > 0 {
		usage = &llmrouter.Usage{
			PromptTokens:     int(chunk.Usage.PromptTokens),
			CompletionTokens: int(chunk.Usage.CompletionTokens),
			TotalTokens:      int(chunk.Usage.TotalTokens),
		}
	}

	return &llmrouter.Response{
		ID:       chunk.ID,
		Object:   string(chunk.Object),
		Created:  chunk.Created,
		Model:    chunk.Model,
		Choices:  choices,
		Usage:    usage,
		Provider: provider,
	}
}

func convertStreamToolCalls(toolCalls []openai.ChatCompletionChunkChoicesDeltaToolCall) []llmrouter.ToolCall {
	result := make([]llmrouter.ToolCall, len(toolCalls))

	for i, tc := range toolCalls {
		idx := int(tc.Index)
		result[i] = llmrouter.ToolCall{
			ID:    tc.ID,
			Type:  "function",
			Index: &idx,
			Function: llmrouter.FuncCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		}
	}

	return result
}

func wrapError(provider string, err error) error {
	if err == nil {
		return nil
	}

	// Try to extract API error details
	apiErr := &llmrouter.APIError{
		Provider: provider,
		Message:  err.Error(),
		Err:      err,
	}

	// Check for OpenAI-specific error types
	if oaiErr, ok := err.(*openai.Error); ok {
		apiErr.StatusCode = oaiErr.StatusCode
		apiErr.Message = oaiErr.Message
		apiErr.Type = oaiErr.Type

		switch oaiErr.StatusCode {
		case http.StatusUnauthorized, http.StatusForbidden:
			apiErr.Err = llmrouter.ErrAuthFailed
		case http.StatusTooManyRequests:
			apiErr.Err = llmrouter.ErrRateLimited
		case http.StatusBadRequest:
			apiErr.Err = llmrouter.ErrInvalidRequest
		}
	}

	return apiErr
}

// Helper to get current time for responses
func now() int64 {
	return time.Now().Unix()
}
