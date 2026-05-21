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

package openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	llmrouter "github.com/bluefunda/llmrouter"
)

func TestNew_CustomHeaders(t *testing.T) {
	// Track headers received by the mock server
	var receivedHeaders http.Header

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHeaders = r.Header.Clone()
		// Return a minimal valid chat completion response
		resp := map[string]interface{}{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1234567890,
			"model":   "sarvam-m",
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"message":       map[string]string{"role": "assistant", "content": "hello"},
					"finish_reason": "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := New(llmrouter.ProviderConfig{
		Name:    "sarvam",
		BaseURL: server.URL,
		APIKey:  "unused",
		CustomHeaders: map[string]string{
			"api-subscription-key": "test-sarvam-key",
		},
	})

	// Make a request to trigger the custom headers
	_, err := provider.Complete(t.Context(), &llmrouter.Request{
		Model: "sarvam-m",
		Messages: []llmrouter.Message{
			{Role: llmrouter.RoleUser, Content: "hi"},
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify custom header was sent
	if got := receivedHeaders.Get("Api-Subscription-Key"); got != "test-sarvam-key" {
		t.Errorf("expected api-subscription-key header %q, got %q", "test-sarvam-key", got)
	}
}

func TestNew_SarvamPreset(t *testing.T) {
	provider := New(llmrouter.ProviderConfig{
		Name:   "sarvam",
		APIKey: "test-key",
	})

	if provider.Name() != "sarvam" {
		t.Errorf("expected name %q, got %q", "sarvam", provider.Name())
	}

	models := provider.Models()
	if len(models) != 3 || models[0] != "sarvam-m" {
		t.Errorf("expected models [sarvam-m sarvam-30b sarvam-105b], got %v", models)
	}
}

func TestNew_NilCustomHeaders(t *testing.T) {
	// Ensure nil CustomHeaders doesn't cause issues
	provider := New(llmrouter.ProviderConfig{
		Name:   "openai",
		APIKey: "test-key",
	})

	if provider.Name() != "openai" {
		t.Errorf("expected name %q, got %q", "openai", provider.Name())
	}
}
