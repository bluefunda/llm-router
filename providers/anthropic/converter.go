package anthropic

import (
	"encoding/json"
	"net/http"

	llmrouter "github.com/bluefunda/llm-router"
	"github.com/anthropics/anthropic-sdk-go"
)

// convertMessages converts llmrouter messages to Anthropic format
// Returns the messages and the system prompt (extracted from system messages)
func convertMessages(msgs []llmrouter.Message) ([]anthropic.MessageParam, string) {
	var systemPrompt string
	var messages []anthropic.MessageParam

	for _, msg := range msgs {
		switch msg.Role {
		case llmrouter.RoleSystem:
			// Anthropic handles system prompts separately
			if systemPrompt != "" {
				systemPrompt += "\n\n"
			}
			systemPrompt += msg.Content

		case llmrouter.RoleUser:
			if len(msg.ContentParts) > 0 {
				blocks := []anthropic.ContentBlockParamUnion{}
				for _, p := range msg.ContentParts {
					switch p.Type {
					case "text":
						blocks = append(blocks, anthropic.NewTextBlock(p.Text))
					case "image_url":
						if p.ImageURL != nil && p.ImageURL.Base64 != "" {
							blocks = append(blocks, anthropic.NewImageBlockBase64(
								p.ImageURL.MediaType,
								p.ImageURL.Base64,
							))
						}
					}
				}
				messages = append(messages, anthropic.NewUserMessage(blocks...))
			} else {
				messages = append(messages, anthropic.NewUserMessage(
					anthropic.NewTextBlock(msg.Content),
				))
			}

		case llmrouter.RoleAssistant:
			if len(msg.ToolCalls) > 0 {
				// Assistant message with tool calls
				blocks := []anthropic.ContentBlockParamUnion{}
				if msg.Content != "" {
					blocks = append(blocks, anthropic.NewTextBlock(msg.Content))
				}
				for _, tc := range msg.ToolCalls {
					var input map[string]interface{}
					_ = json.Unmarshal([]byte(tc.Function.Arguments), &input)
					blocks = append(blocks, anthropic.NewToolUseBlockParam(tc.ID, tc.Function.Name, input))
				}
				messages = append(messages, anthropic.NewAssistantMessage(blocks...))
			} else {
				messages = append(messages, anthropic.NewAssistantMessage(
					anthropic.NewTextBlock(msg.Content),
				))
			}

		case llmrouter.RoleTool:
			// Tool result message
			messages = append(messages, anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(msg.ToolCallID, msg.Content, false),
			))
		}
	}

	return messages, systemPrompt
}

// convertTools converts llmrouter tools to Anthropic format
func convertTools(tools []llmrouter.Tool) []anthropic.ToolParam {
	result := make([]anthropic.ToolParam, len(tools))

	for i, tool := range tools {
		// Parse the parameters JSON schema
		var inputSchema interface{}
		if tool.Function.Parameters != nil {
			var params map[string]interface{}
			_ = json.Unmarshal(tool.Function.Parameters, &params)
			// Ensure type is set to object
			if params != nil {
				params["type"] = "object"
				inputSchema = params
			}
		}
		if inputSchema == nil {
			inputSchema = map[string]interface{}{"type": "object"}
		}

		result[i] = anthropic.ToolParam{
			Name:        anthropic.F(tool.Function.Name),
			Description: anthropic.F(tool.Function.Description),
			InputSchema: anthropic.F(inputSchema),
		}
	}

	return result
}

// convertToolChoice converts llmrouter tool choice to Anthropic format
func convertToolChoice(tc *llmrouter.ToolChoice) anthropic.ToolChoiceUnionParam {
	if tc == nil {
		return nil
	}

	switch tc.Type {
	case "auto":
		return anthropic.ToolChoiceAutoParam{
			Type: anthropic.F(anthropic.ToolChoiceAutoTypeAuto),
		}
	case "none":
		// Anthropic doesn't have "none" - use auto as fallback
		return anthropic.ToolChoiceAutoParam{
			Type: anthropic.F(anthropic.ToolChoiceAutoTypeAuto),
		}
	case "required", "any":
		return anthropic.ToolChoiceAnyParam{
			Type: anthropic.F(anthropic.ToolChoiceAnyTypeAny),
		}
	case "function":
		if tc.Function != nil {
			return anthropic.ToolChoiceToolParam{
				Type: anthropic.F(anthropic.ToolChoiceToolTypeTool),
				Name: anthropic.F(tc.Function.Name),
			}
		}
	}

	return nil
}

// convertToOpenAIResponse converts Anthropic response to OpenAI-compatible format
func convertToOpenAIResponse(msg *anthropic.Message, provider string) *llmrouter.Response {
	var content string
	var toolCalls []llmrouter.ToolCall

	for _, block := range msg.Content {
		switch b := block.AsUnion().(type) {
		case anthropic.TextBlock:
			content += b.Text
		case anthropic.ToolUseBlock:
			args, _ := json.Marshal(b.Input)
			toolCalls = append(toolCalls, llmrouter.ToolCall{
				ID:   b.ID,
				Type: "function",
				Function: llmrouter.FuncCall{
					Name:      b.Name,
					Arguments: string(args),
				},
			})
		}
	}

	finishReason := "stop"
	switch msg.StopReason {
	case anthropic.MessageStopReasonToolUse:
		finishReason = "tool_calls"
	case anthropic.MessageStopReasonMaxTokens:
		finishReason = "length"
	case anthropic.MessageStopReasonStopSequence:
		finishReason = "stop"
	}

	return &llmrouter.Response{
		ID:       msg.ID,
		Object:   "chat.completion",
		Model:    string(msg.Model),
		Provider: provider,
		Choices: []llmrouter.Choice{
			{
				Index: 0,
				Message: &llmrouter.Message{
					Role:      llmrouter.RoleAssistant,
					Content:   content,
					ToolCalls: toolCalls,
				},
				FinishReason: finishReason,
			},
		},
		Usage: &llmrouter.Usage{
			PromptTokens:     int(msg.Usage.InputTokens),
			CompletionTokens: int(msg.Usage.OutputTokens),
			TotalTokens:      int(msg.Usage.InputTokens + msg.Usage.OutputTokens),
		},
	}
}

// wrapError wraps Anthropic errors
func wrapError(err error) error {
	if err == nil {
		return nil
	}

	apiErr := &llmrouter.APIError{
		Provider: "anthropic",
		Message:  err.Error(),
		Err:      err,
	}

	// Check for Anthropic-specific error types
	if antErr, ok := err.(*anthropic.Error); ok {
		apiErr.StatusCode = antErr.StatusCode

		switch antErr.StatusCode {
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
