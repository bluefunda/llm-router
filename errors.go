package llmrouter

import (
	"errors"
	"net/http"
)

// Sentinel errors
var (
	ErrUnknownModel     = errors.New("unknown model")
	ErrUnknownProvider  = errors.New("unknown provider")
	ErrNoProviders      = errors.New("no providers registered")
	ErrRateLimited      = errors.New("rate limited")
	ErrContextCanceled  = errors.New("context canceled")
	ErrStreamClosed     = errors.New("stream closed")
	ErrInvalidRequest   = errors.New("invalid request")
	ErrAuthFailed       = errors.New("authentication failed")
	ErrProviderError    = errors.New("provider error")
	ErrCircuitOpen      = errors.New("circuit breaker is open")
	ErrMaxRetriesExceed = errors.New("max retries exceeded")
)

// APIError represents an error from an LLM provider API
type APIError struct {
	Provider   string
	StatusCode int
	Message    string
	Type       string
	Err        error
}

func (e *APIError) Error() string {
	if e.Err != nil {
		return e.Provider + ": " + e.Message + ": " + e.Err.Error()
	}
	return e.Provider + ": " + e.Message
}

func (e *APIError) Unwrap() error {
	return e.Err
}

// IsRetryable returns true if the error is retryable
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}

	// Check for context cancellation
	if errors.Is(err, ErrContextCanceled) {
		return false
	}

	// Check for auth errors - not retryable
	if errors.Is(err, ErrAuthFailed) {
		return false
	}

	// Check for invalid request - not retryable
	if errors.Is(err, ErrInvalidRequest) {
		return false
	}

	// Check API errors
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		switch apiErr.StatusCode {
		case http.StatusTooManyRequests: // 429 - rate limited, retryable
			return true
		case http.StatusInternalServerError, // 500
			http.StatusBadGateway,         // 502
			http.StatusServiceUnavailable, // 503
			http.StatusGatewayTimeout:     // 504
			return true
		case http.StatusUnauthorized, // 401
			http.StatusForbidden,  // 403
			http.StatusBadRequest: // 400
			return false
		}
	}

	// Rate limit errors are retryable
	if errors.Is(err, ErrRateLimited) {
		return true
	}

	// Default to retryable for unknown errors
	return true
}

// IsRateLimited returns true if the error indicates rate limiting
func IsRateLimited(err error) bool {
	if errors.Is(err, ErrRateLimited) {
		return true
	}

	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.StatusCode == http.StatusTooManyRequests
	}

	return false
}
