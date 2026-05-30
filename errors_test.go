package llmrouter_test

import (
	"errors"
	"testing"

	llmrouter "github.com/bluefunda/llmrouter"
)

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{
			name: "nil error",
			err:  nil,
			want: false,
		},
		{
			name: "status 429 rate limited",
			err:  &llmrouter.APIError{StatusCode: 429, Message: "rate limited"},
			want: true,
		},
		{
			name: "status 500 internal server error",
			err:  &llmrouter.APIError{StatusCode: 500, Message: "internal server error"},
			want: true,
		},
		{
			name: "status 502 bad gateway",
			err:  &llmrouter.APIError{StatusCode: 502, Message: "bad gateway"},
			want: true,
		},
		{
			name: "status 503 service unavailable",
			err:  &llmrouter.APIError{StatusCode: 503, Message: "service unavailable"},
			want: true,
		},
		{
			name: "status 504 gateway timeout",
			err:  &llmrouter.APIError{StatusCode: 504, Message: "gateway timeout"},
			want: true,
		},
		{
			name: "status 400 bad request",
			err:  &llmrouter.APIError{StatusCode: 400, Message: "bad request"},
			want: false,
		},
		{
			name: "status 401 unauthorized",
			err:  &llmrouter.APIError{StatusCode: 401, Message: "unauthorized"},
			want: false,
		},
		{
			name: "status 403 forbidden",
			err:  &llmrouter.APIError{StatusCode: 403, Message: "forbidden"},
			want: false,
		},
		{
			name: "status 404 not found",
			err:  &llmrouter.APIError{StatusCode: 404, Message: "not found"},
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := llmrouter.IsRetryable(tc.err)
			if got != tc.want {
				t.Errorf("IsRetryable(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestIsRateLimited(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{
			name: "nil error",
			err:  nil,
			want: false,
		},
		{
			name: "status 429",
			err:  &llmrouter.APIError{StatusCode: 429, Message: "rate limited"},
			want: true,
		},
		{
			name: "ErrRateLimited sentinel",
			err:  llmrouter.ErrRateLimited,
			want: true,
		},
		{
			name: "status 500",
			err:  &llmrouter.APIError{StatusCode: 500, Message: "server error"},
			want: false,
		},
		{
			name: "status 400",
			err:  &llmrouter.APIError{StatusCode: 400, Message: "bad request"},
			want: false,
		},
		{
			name: "ErrProviderError",
			err:  llmrouter.ErrProviderError,
			want: false,
		},
		{
			name: "ErrUnknownModel",
			err:  llmrouter.ErrUnknownModel,
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := llmrouter.IsRateLimited(tc.err)
			if got != tc.want {
				t.Errorf("IsRateLimited(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestAPIError_Is(t *testing.T) {
	apiErr := &llmrouter.APIError{
		Provider:   "test-provider",
		StatusCode: 429,
		Message:    "too many requests",
		Err:        llmrouter.ErrRateLimited,
	}

	if !errors.Is(apiErr, llmrouter.ErrRateLimited) {
		t.Error("expected errors.Is(apiErr, ErrRateLimited) to be true via Unwrap")
	}

	if errors.Is(apiErr, llmrouter.ErrUnknownModel) {
		t.Error("expected errors.Is(apiErr, ErrUnknownModel) to be false")
	}
}

func TestErrUnknownModel_Sentinel(t *testing.T) {
	wrapped := llmrouter.ErrUnknownModel

	if !errors.Is(wrapped, llmrouter.ErrUnknownModel) {
		t.Error("expected errors.Is to find ErrUnknownModel")
	}

	// Wrap it like the router does and verify it still unwraps.
	wrappedFmt := errors.Join(llmrouter.ErrUnknownModel, errors.New("extra context"))
	if !errors.Is(wrappedFmt, llmrouter.ErrUnknownModel) {
		t.Error("expected errors.Is to find ErrUnknownModel through errors.Join wrapping")
	}
}
