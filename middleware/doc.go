// Package middleware provides composable cross-cutting concerns for LLM
// provider calls: retry with exponential backoff, per-request timeouts, and
// a circuit breaker to prevent cascading failures.
//
// Middleware wraps any [llmrouter.Provider] and is applied in declaration
// order — the first middleware registered is the outermost wrapper.
//
// # Retry
//
//	mw := middleware.Retry(3, time.Second)
//	mw := middleware.Retry(3, time.Second, middleware.WithMaxDelay(10*time.Second))
//
// Non-retryable errors (auth failures, invalid requests, context
// cancellation) short-circuit immediately without consuming retry attempts.
//
// # Circuit breaker
//
//	cb := middleware.NewCircuitBreaker(5, 30*time.Second)
//	// pass cb.Wrap where a MiddlewareFunc is expected
//	router := llmrouter.New(llmrouter.WithMiddleware(cb.Wrap))
//	// inspect state at any time
//	fmt.Println(cb.State())
//
// Opens after consecutive failures exceed the threshold; recovers after the
// timeout elapses. Stdlib-only — no external dependencies.
//
// # Timeout
//
//	mw := middleware.Timeout(60 * time.Second)
//
// Enforces a deadline on both Complete and Stream calls. On timeout,
// Stream returns an error via StreamResult.Err() rather than blocking indefinitely.
package middleware
