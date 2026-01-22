package gemini

import (
	"encoding/json"
	"time"

	llmrouter "github.com/bluefunda/llm-router"
	"github.com/google/generative-ai-go/genai"
)

// convertHistory converts llmrouter messages to Gemini chat history
// Returns the history and the last user message (which should be sent separately)
func convertHistory(msgs []llmrouter.Message) ([]*genai.Content, string) {
	var history []*genai.Content
	var lastUserMsg string

	for i, msg := range msgs {
		switch msg.Role {
		case llmrouter.RoleSystem:
			// System messages are handled separately via SystemInstruction
			continue

		case llmrouter.RoleUser:
			// If this is the last message, save it for sending
			if i == len(msgs)-1 {
				lastUserMsg = msg.Content
				continue
			}
			history = append(history, &genai.Content{
				Role:  "user",
				Parts: []genai.Part{genai.Text(msg.Content)},
			})

		case llmrouter.RoleAssistant:
			parts := []genai.Part{}
			if msg.Content != "" {
				parts = append(parts, genai.Text(msg.Content))
			}
			for _, tc := range msg.ToolCalls {
				var args map[string]interface{}
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				parts = append(parts, genai.FunctionCall{
					Name: tc.Function.Name,
					Args: args,
				})
			}
			if len(parts) > 0 {
				history = append(history, &genai.Content{
					Role:  "model",
					Parts: parts,
				})
			}

		case llmrouter.RoleTool:
			// Tool results
			var result map[string]interface{}
			_ = json.Unmarshal([]byte(msg.Content), &result)
			if result == nil {
				result = map[string]interface{}{"result": msg.Content}
			}
			history = append(history, &genai.Content{
				Role: "function",
				Parts: []genai.Part{
					genai.FunctionResponse{
						Name:     msg.Name,
						Response: result,
					},
				},
			})
		}
	}

	return history, lastUserMsg
}

// convertTools converts llmrouter tools to Gemini format
func convertTools(tools []llmrouter.Tool) []*genai.Tool {
	funcDecls := make([]*genai.FunctionDeclaration, len(tools))

	for i, tool := range tools {
		var schema *genai.Schema
		if tool.Function.Parameters != nil {
			var params map[string]interface{}
			_ = json.Unmarshal(tool.Function.Parameters, &params)
			schema = convertSchema(params)
		}

		funcDecls[i] = &genai.FunctionDeclaration{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			Parameters:  schema,
		}
	}

	return []*genai.Tool{
		{FunctionDeclarations: funcDecls},
	}
}

// convertSchema converts a JSON schema to Gemini Schema
func convertSchema(params map[string]interface{}) *genai.Schema {
	if params == nil {
		return nil
	}

	schema := &genai.Schema{
		Type: genai.TypeObject,
	}

	if props, ok := params["properties"].(map[string]interface{}); ok {
		schema.Properties = make(map[string]*genai.Schema)
		for name, prop := range props {
			if propMap, ok := prop.(map[string]interface{}); ok {
				schema.Properties[name] = convertPropertySchema(propMap)
			}
		}
	}

	if required, ok := params["required"].([]interface{}); ok {
		for _, r := range required {
			if s, ok := r.(string); ok {
				schema.Required = append(schema.Required, s)
			}
		}
	}

	return schema
}

func convertPropertySchema(prop map[string]interface{}) *genai.Schema {
	schema := &genai.Schema{}

	if t, ok := prop["type"].(string); ok {
		switch t {
		case "string":
			schema.Type = genai.TypeString
		case "number":
			schema.Type = genai.TypeNumber
		case "integer":
			schema.Type = genai.TypeInteger
		case "boolean":
			schema.Type = genai.TypeBoolean
		case "array":
			schema.Type = genai.TypeArray
			if items, ok := prop["items"].(map[string]interface{}); ok {
				schema.Items = convertPropertySchema(items)
			}
		case "object":
			schema.Type = genai.TypeObject
			if props, ok := prop["properties"].(map[string]interface{}); ok {
				schema.Properties = make(map[string]*genai.Schema)
				for name, p := range props {
					if pm, ok := p.(map[string]interface{}); ok {
						schema.Properties[name] = convertPropertySchema(pm)
					}
				}
			}
		}
	}

	if desc, ok := prop["description"].(string); ok {
		schema.Description = desc
	}

	if enum, ok := prop["enum"].([]interface{}); ok {
		for _, e := range enum {
			if s, ok := e.(string); ok {
				schema.Enum = append(schema.Enum, s)
			}
		}
	}

	return schema
}

// convertResponse converts Gemini response to OpenAI-compatible format
func convertResponse(resp *genai.GenerateContentResponse, model, provider string) *llmrouter.Response {
	var content string
	var toolCalls []llmrouter.ToolCall

	if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
		for _, part := range resp.Candidates[0].Content.Parts {
			switch p := part.(type) {
			case genai.Text:
				content += string(p)
			case genai.FunctionCall:
				args, _ := convertFunctionCallArgs(p.Args)
				toolCalls = append(toolCalls, llmrouter.ToolCall{
					ID:   p.Name,
					Type: "function",
					Function: llmrouter.FuncCall{
						Name:      p.Name,
						Arguments: args,
					},
				})
			}
		}
	}

	finishReason := "stop"
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	} else if len(resp.Candidates) > 0 {
		switch resp.Candidates[0].FinishReason {
		case genai.FinishReasonMaxTokens:
			finishReason = "length"
		case genai.FinishReasonStop:
			finishReason = "stop"
		case genai.FinishReasonSafety:
			finishReason = "content_filter"
		}
	}

	var usage *llmrouter.Usage
	if resp.UsageMetadata != nil {
		usage = &llmrouter.Usage{
			PromptTokens:     int(resp.UsageMetadata.PromptTokenCount),
			CompletionTokens: int(resp.UsageMetadata.CandidatesTokenCount),
			TotalTokens:      int(resp.UsageMetadata.TotalTokenCount),
		}
	}

	return &llmrouter.Response{
		Model:    model,
		Provider: provider,
		Object:   "chat.completion",
		Created:  time.Now().Unix(),
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
		Usage: usage,
	}
}

// convertFunctionCallArgs converts function call args to JSON string
func convertFunctionCallArgs(args map[string]interface{}) (string, error) {
	if args == nil {
		return "{}", nil
	}
	b, err := json.Marshal(args)
	return string(b), err
}

// wrapError wraps Gemini errors
func wrapError(err error) error {
	if err == nil {
		return nil
	}

	return &llmrouter.APIError{
		Provider: "gemini",
		Message:  err.Error(),
		Err:      err,
	}
}
