package openai

import (
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	llmrouter "github.com/bluefunda/llmrouter"
	oai "github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

func TestConvertMessages_System(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleSystem, Content: "you are helpful"},
	}
	result := convertMessages(msgs, false)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].OfSystem == nil {
		t.Fatal("expected system message")
	}
	got := result[0].OfSystem.Content.OfString.Value
	if got != "you are helpful" {
		t.Errorf("expected 'you are helpful', got %q", got)
	}
}

func TestConvertMessages_UserPlainText(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleUser, Content: "hello"},
	}
	result := convertMessages(msgs, false)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].OfUser == nil {
		t.Fatal("expected user message")
	}
}

func TestConvertMessages_UserStringOnly(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleUser, ContentParts: []llmrouter.ContentPart{
			{Type: "text", Text: "hi"},
		}},
	}
	result := convertMessages(msgs, true)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].OfUser == nil {
		t.Fatal("expected user message")
	}
}

func TestConvertMessages_UserMultiPart(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleUser, ContentParts: []llmrouter.ContentPart{
			{Type: "text", Text: "describe this"},
			{Type: "image_url", ImageURL: &llmrouter.ImageURL{URL: "https://example.com/img.png"}},
		}},
	}
	result := convertMessages(msgs, false)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].OfUser == nil {
		t.Fatal("expected user message")
	}
}

func TestConvertMessages_UserImageURLNilSkipped(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleUser, ContentParts: []llmrouter.ContentPart{
			{Type: "text", Text: "hello"},
			{Type: "image_url", ImageURL: nil},
		}},
	}
	result := convertMessages(msgs, false)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
}

func TestConvertMessages_AssistantNoTools(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleAssistant, Content: "sure, I can help"},
	}
	result := convertMessages(msgs, false)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].OfAssistant == nil {
		t.Fatal("expected assistant message")
	}
}

func TestConvertMessages_AssistantWithTools(t *testing.T) {
	msgs := []llmrouter.Message{
		{
			Role:    llmrouter.RoleAssistant,
			Content: "calling tool",
			ToolCalls: []llmrouter.ToolCall{
				{
					ID:   "call_abc",
					Type: "function",
					Function: llmrouter.FuncCall{
						Name:      "get_weather",
						Arguments: `{"location":"NYC"}`,
					},
				},
			},
		},
	}
	result := convertMessages(msgs, false)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].OfAssistant == nil {
		t.Fatal("expected assistant message")
	}
	if len(result[0].OfAssistant.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result[0].OfAssistant.ToolCalls))
	}
	tc := result[0].OfAssistant.ToolCalls[0]
	if tc.ID != "call_abc" {
		t.Errorf("expected ID call_abc, got %s", tc.ID)
	}
	if tc.Function.Name != "get_weather" {
		t.Errorf("expected function name get_weather, got %s", tc.Function.Name)
	}
}

func TestConvertMessages_AssistantWithToolsNoContent(t *testing.T) {
	msgs := []llmrouter.Message{
		{
			Role: llmrouter.RoleAssistant,
			ToolCalls: []llmrouter.ToolCall{
				{ID: "call_1", Function: llmrouter.FuncCall{Name: "f", Arguments: "{}"}},
			},
		},
	}
	result := convertMessages(msgs, false)
	if len(result) != 1 {
		t.Fatalf("expected 1, got %d", len(result))
	}
	if result[0].OfAssistant == nil {
		t.Fatal("expected assistant")
	}
}

func TestConvertMessages_ToolRole(t *testing.T) {
	msgs := []llmrouter.Message{
		{Role: llmrouter.RoleTool, Content: `{"temp":72}`, ToolCallID: "call_abc"},
	}
	result := convertMessages(msgs, false)
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].OfTool == nil {
		t.Fatal("expected tool message")
	}
	if result[0].OfTool.ToolCallID != "call_abc" {
		t.Errorf("expected ToolCallID call_abc, got %s", result[0].OfTool.ToolCallID)
	}
}

func TestConvertTools_Basic(t *testing.T) {
	raw := json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`)
	tools := []llmrouter.Tool{
		{
			Type: "function",
			Function: llmrouter.Function{
				Name:        "search",
				Description: "search the web",
				Parameters:  raw,
			},
		},
	}
	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].Function.Name != "search" {
		t.Errorf("expected name search, got %s", result[0].Function.Name)
	}
	if !result[0].Function.Description.Valid() || result[0].Function.Description.Value != "search the web" {
		t.Errorf("unexpected description: %v", result[0].Function.Description)
	}
}

func TestConvertTools_NilParameters(t *testing.T) {
	tools := []llmrouter.Tool{
		{
			Function: llmrouter.Function{
				Name: "noop",
			},
		},
	}
	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
}

func TestConvertToolChoice_Auto(t *testing.T) {
	tc := &llmrouter.ToolChoice{Type: "auto"}
	result := convertToolChoice(tc)
	if !result.OfAuto.Valid() || result.OfAuto.Value != "auto" {
		t.Errorf("expected auto, got %+v", result)
	}
}

func TestConvertToolChoice_None(t *testing.T) {
	tc := &llmrouter.ToolChoice{Type: "none"}
	result := convertToolChoice(tc)
	if !result.OfAuto.Valid() || result.OfAuto.Value != "none" {
		t.Errorf("expected none, got %+v", result)
	}
}

func TestConvertToolChoice_Required(t *testing.T) {
	tc := &llmrouter.ToolChoice{Type: "required"}
	result := convertToolChoice(tc)
	if !result.OfAuto.Valid() || result.OfAuto.Value != "required" {
		t.Errorf("expected required, got %+v", result)
	}
}

func TestConvertToolChoice_Function(t *testing.T) {
	tc := &llmrouter.ToolChoice{Type: "function", Function: &llmrouter.FuncRef{Name: "get_weather"}}
	result := convertToolChoice(tc)
	if result.OfChatCompletionNamedToolChoice == nil {
		t.Fatal("expected named tool choice")
	}
	if result.OfChatCompletionNamedToolChoice.Function.Name != "get_weather" {
		t.Errorf("expected get_weather, got %s", result.OfChatCompletionNamedToolChoice.Function.Name)
	}
}

func TestConvertToolChoice_FunctionNilRef(t *testing.T) {
	tc := &llmrouter.ToolChoice{Type: "function", Function: nil}
	result := convertToolChoice(tc)
	if result.OfChatCompletionNamedToolChoice != nil {
		t.Error("expected empty result when function ref is nil")
	}
	if param.IsOmitted(result.OfAuto) == false {
		t.Error("expected OfAuto to be omitted")
	}
}

func TestConvertToolChoice_Nil(t *testing.T) {
	result := convertToolChoice(nil)
	_ = result // should not panic
}

func TestConvertStreamToolCalls(t *testing.T) {
	idx := int64(0)
	calls := []oai.ChatCompletionChunkChoiceDeltaToolCall{
		{
			Index: idx,
			ID:    "call_1",
			Function: oai.ChatCompletionChunkChoiceDeltaToolCallFunction{
				Name:      "get_weather",
				Arguments: `{"loc":`,
			},
		},
	}
	result := convertStreamToolCalls(calls)
	if len(result) != 1 {
		t.Fatalf("expected 1, got %d", len(result))
	}
	if result[0].ID != "call_1" {
		t.Errorf("expected ID call_1, got %s", result[0].ID)
	}
	if result[0].Function.Name != "get_weather" {
		t.Errorf("expected get_weather, got %s", result[0].Function.Name)
	}
	if result[0].Index == nil || *result[0].Index != 0 {
		t.Errorf("expected index 0, got %v", result[0].Index)
	}
}

func TestWrapError_Nil(t *testing.T) {
	if wrapError("openai", nil) != nil {
		t.Error("expected nil")
	}
}

func TestWrapError_Generic(t *testing.T) {
	err := errors.New("something went wrong")
	wrapped := wrapError("openai", err)
	var apiErr *llmrouter.APIError
	if !errors.As(wrapped, &apiErr) {
		t.Fatal("expected APIError")
	}
	if apiErr.Provider != "openai" {
		t.Errorf("expected provider openai, got %s", apiErr.Provider)
	}
}

func makeOAIError(statusCode int) *oai.Error {
	req, _ := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", nil)
	resp := &http.Response{StatusCode: statusCode, Request: req}
	return &oai.Error{StatusCode: statusCode, Request: req, Response: resp}
}

func TestWrapError_RateLimited(t *testing.T) {
	wrapped := wrapError("openai", makeOAIError(http.StatusTooManyRequests))
	if !errors.Is(wrapped, llmrouter.ErrRateLimited) {
		t.Errorf("expected ErrRateLimited, got %v", wrapped)
	}
}

func TestWrapError_AuthFailed_401(t *testing.T) {
	wrapped := wrapError("openai", makeOAIError(http.StatusUnauthorized))
	if !errors.Is(wrapped, llmrouter.ErrAuthFailed) {
		t.Errorf("expected ErrAuthFailed, got %v", wrapped)
	}
}

func TestWrapError_AuthFailed_403(t *testing.T) {
	wrapped := wrapError("openai", makeOAIError(http.StatusForbidden))
	if !errors.Is(wrapped, llmrouter.ErrAuthFailed) {
		t.Errorf("expected ErrAuthFailed, got %v", wrapped)
	}
}

func TestWrapError_InvalidRequest(t *testing.T) {
	wrapped := wrapError("openai", makeOAIError(http.StatusBadRequest))
	if !errors.Is(wrapped, llmrouter.ErrInvalidRequest) {
		t.Errorf("expected ErrInvalidRequest, got %v", wrapped)
	}
}

func TestConvertResponse_Basic(t *testing.T) {
	resp := &oai.ChatCompletion{
		ID:    "chatcmpl-123",
		Model: "gpt-4o",
		Choices: []oai.ChatCompletionChoice{
			{
				Index: 0,
				Message: oai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "Hello!",
				},
				FinishReason: "stop",
			},
		},
		Usage: oai.CompletionUsage{
			PromptTokens:     10,
			CompletionTokens: 5,
			TotalTokens:      15,
		},
	}
	result := convertResponse(resp, "openai")
	if result.ID != "chatcmpl-123" {
		t.Errorf("unexpected ID: %s", result.ID)
	}
	if result.Provider != "openai" {
		t.Errorf("unexpected provider: %s", result.Provider)
	}
	if len(result.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(result.Choices))
	}
	if result.Choices[0].Message.Content != "Hello!" {
		t.Errorf("unexpected content: %s", result.Choices[0].Message.Content)
	}
	if result.Usage == nil {
		t.Fatal("expected usage")
	}
	if result.Usage.TotalTokens != 15 {
		t.Errorf("expected 15 total tokens, got %d", result.Usage.TotalTokens)
	}
}

func TestConvertResponse_WithToolCalls(t *testing.T) {
	resp := &oai.ChatCompletion{
		ID:    "chatcmpl-456",
		Model: "gpt-4o",
		Choices: []oai.ChatCompletionChoice{
			{
				Index: 0,
				Message: oai.ChatCompletionMessage{
					Role: "assistant",
					ToolCalls: []oai.ChatCompletionMessageToolCall{
						{
							ID:   "call_xyz",
							Type: "function",
							Function: oai.ChatCompletionMessageToolCallFunction{
								Name:      "search",
								Arguments: `{"q":"go lang"}`,
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
		Usage: oai.CompletionUsage{TotalTokens: 20},
	}
	result := convertResponse(resp, "openai")
	if len(result.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(result.Choices[0].Message.ToolCalls))
	}
	if result.Choices[0].Message.ToolCalls[0].ID != "call_xyz" {
		t.Errorf("unexpected tool call ID: %s", result.Choices[0].Message.ToolCalls[0].ID)
	}
}

func TestConvertResponse_NoUsage(t *testing.T) {
	resp := &oai.ChatCompletion{
		ID:    "chatcmpl-789",
		Model: "gpt-4o",
		Choices: []oai.ChatCompletionChoice{
			{
				Index:        0,
				Message:      oai.ChatCompletionMessage{Role: "assistant", Content: "hi"},
				FinishReason: "stop",
			},
		},
	}
	result := convertResponse(resp, "openai")
	if result.Usage != nil {
		t.Error("expected nil usage when TotalTokens is 0")
	}
}
