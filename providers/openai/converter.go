package openai

import (
	"encoding/json"
	"net/http"

	llmrouter "github.com/bluefunda/llmrouter"
	"github.com/openai/openai-go"
)

// convertMessages converts llmrouter messages to the OpenAI SDK type.
// When stringOnly is true, single-text-part user messages are sent as plain
// strings rather than content arrays, for providers that require this format.
func convertMessages(msgs []llmrouter.Message, stringOnly bool) []openai.ChatCompletionMessageParamUnion {
	result := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs))

	for _, msg := range msgs {
		switch msg.Role {
		case llmrouter.RoleSystem:
			result = append(result, openai.SystemMessage(msg.Content))

		case llmrouter.RoleUser:
			if len(msg.ContentParts) > 0 {
				if stringOnly && len(msg.ContentParts) == 1 && msg.ContentParts[0].Type == "text" {
					result = append(result, openai.UserMessage(msg.ContentParts[0].Text))
				} else {
					parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(msg.ContentParts))
					for _, p := range msg.ContentParts {
						switch p.Type {
						case "text":
							parts = append(parts, openai.TextContentPart(p.Text))
						case "image_url":
							if p.ImageURL != nil {
								parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{URL: p.ImageURL.URL}))
							}
						}
					}
					result = append(result, openai.UserMessage(parts))
				}
			} else {
				result = append(result, openai.UserMessage(msg.Content))
			}

		case llmrouter.RoleAssistant:
			if len(msg.ToolCalls) > 0 {
				toolCalls := make([]openai.ChatCompletionMessageToolCallParam, len(msg.ToolCalls))
				for i, tc := range msg.ToolCalls {
					toolCalls[i] = openai.ChatCompletionMessageToolCallParam{
						ID: tc.ID,
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      tc.Function.Name,
							Arguments: tc.Function.Arguments,
						},
					}
				}
				assistant := openai.ChatCompletionAssistantMessageParam{
					ToolCalls: toolCalls,
				}
				if msg.Content != "" {
					assistant.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: openai.String(msg.Content),
					}
				}
				result = append(result, openai.ChatCompletionMessageParamUnion{OfAssistant: &assistant})
			} else {
				result = append(result, openai.AssistantMessage(msg.Content))
			}

		case llmrouter.RoleTool:
			result = append(result, openai.ToolMessage(msg.Content, msg.ToolCallID))
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
			Function: openai.FunctionDefinitionParam{
				Name:        tool.Function.Name,
				Description: openai.String(tool.Function.Description),
				Parameters:  openai.FunctionParameters(params),
			},
		}
	}

	return result
}

func convertToolChoice(tc *llmrouter.ToolChoice) openai.ChatCompletionToolChoiceOptionUnionParam {
	if tc == nil {
		return openai.ChatCompletionToolChoiceOptionUnionParam{}
	}

	switch tc.Type {
	case "auto":
		return openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("auto")}
	case "none":
		return openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("none")}
	case "required":
		return openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: openai.String("required")}
	case "function":
		if tc.Function != nil {
			return openai.ChatCompletionToolChoiceOptionParamOfChatCompletionNamedToolChoice(
				openai.ChatCompletionNamedToolChoiceFunctionParam{Name: tc.Function.Name},
			)
		}
	}

	return openai.ChatCompletionToolChoiceOptionUnionParam{}
}

// deepSeekUsageExtra captures DeepSeek's non-standard context-cache usage
// fields. DeepSeek's disk-based context caching is automatic and reports
// cache hits as top-level usage.prompt_cache_hit_tokens, rather than the
// OpenAI-standard usage.prompt_tokens_details.cached_tokens.
type deepSeekUsageExtra struct {
	PromptCacheHitTokens int64 `json:"prompt_cache_hit_tokens"`
}

// cachedPromptTokens extracts the cached prefix token count from a chat
// completion's usage object. It checks the OpenAI-standard field first, then
// falls back to DeepSeek's prompt_cache_hit_tokens field for providers (like
// DeepSeek) that report cache hits under a different name.
func cachedPromptTokens(usage openai.CompletionUsage) int {
	if usage.PromptTokensDetails.CachedTokens > 0 {
		return int(usage.PromptTokensDetails.CachedTokens)
	}

	raw := usage.RawJSON()
	if raw == "" {
		return 0
	}

	var extra deepSeekUsageExtra
	if err := json.Unmarshal([]byte(raw), &extra); err != nil {
		return 0
	}

	return int(extra.PromptCacheHitTokens)
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
			PromptTokens:       int(resp.Usage.PromptTokens),
			CompletionTokens:   int(resp.Usage.CompletionTokens),
			TotalTokens:        int(resp.Usage.TotalTokens),
			CachedPromptTokens: cachedPromptTokens(resp.Usage),
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
			PromptTokens:       int(chunk.Usage.PromptTokens),
			CompletionTokens:   int(chunk.Usage.CompletionTokens),
			TotalTokens:        int(chunk.Usage.TotalTokens),
			CachedPromptTokens: cachedPromptTokens(chunk.Usage),
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

func convertStreamToolCalls(toolCalls []openai.ChatCompletionChunkChoiceDeltaToolCall) []llmrouter.ToolCall {
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

