package llmrouter

import (
	"encoding/json"
	"time"
)

// Request represents a unified LLM request
type Request struct {
	Messages    []Message      `json:"messages"`
	Model       string         `json:"model,omitempty"`
	Tools       []Tool         `json:"tools,omitempty"`
	ToolChoice  *ToolChoice    `json:"tool_choice,omitempty"`
	Temperature *float64       `json:"temperature,omitempty"`
	MaxTokens   *int           `json:"max_tokens,omitempty"`
	TopP        *float64       `json:"top_p,omitempty"`
	Stop        []string       `json:"stop,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role       Role       `json:"role"`
	Content    string     `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// Role represents the message role
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Response represents a unified LLM response (OpenAI-compatible)
type Response struct {
	ID       string   `json:"id"`
	Object   string   `json:"object"`
	Created  int64    `json:"created"`
	Model    string   `json:"model"`
	Choices  []Choice `json:"choices"`
	Usage    *Usage   `json:"usage,omitempty"`
	Provider string   `json:"provider"`
}

// Choice represents a completion choice
type Choice struct {
	Index        int      `json:"index"`
	Message      *Message `json:"message,omitempty"`
	Delta        *Delta   `json:"delta,omitempty"`
	FinishReason string   `json:"finish_reason,omitempty"`
}

// Delta represents streaming content delta
type Delta struct {
	Role      Role       `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// Usage represents token usage
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Event represents a streaming event
type Event struct {
	Type     EventType
	Content  string
	Delta    *Delta
	Response *Response
	Error    error
}

// EventType represents the type of streaming event
type EventType int

const (
	EventContentDelta  EventType = iota // Text content chunk
	EventToolCallDelta                  // Tool call chunk
	EventDone                           // Stream completed
	EventError                          // Error occurred
)

// Tool represents a function/tool definition
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function represents a function definition
type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ToolCall represents a tool invocation
type ToolCall struct {
	ID       string   `json:"id"`
	Type     string   `json:"type"`
	Function FuncCall `json:"function"`
	Index    *int     `json:"index,omitempty"`
}

// FuncCall represents a function call
type FuncCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolChoice controls tool selection
type ToolChoice struct {
	Type     string   `json:"type,omitempty"`
	Function *FuncRef `json:"function,omitempty"`
}

// FuncRef references a specific function
type FuncRef struct {
	Name string `json:"name"`
}

// ProviderConfig holds common configuration for providers
type ProviderConfig struct {
	Name       string
	APIKey     string
	BaseURL    string
	Model      string
	Models     []string
	MaxRetries int
	Timeout    time.Duration
}
