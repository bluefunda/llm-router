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

// Package middleware provides composable cross-cutting concerns for LLM
// provider calls: retry with exponential backoff, per-request timeouts, and
// a circuit breaker to prevent cascading failures.
//
// Middleware wraps any [llmrouter.Provider] and is applied in declaration
// order — the first middleware registered is the outermost wrapper.
//
// # Retry
//
//	mw := middleware.NewRetryMiddleware(3, time.Second)
//	mw.WithMaxDelay(30 * time.Second)
//
// Non-retryable errors (auth failures, invalid requests, context
// cancellation) short-circuit immediately without consuming retry attempts.
//
// # Circuit breaker
//
//	mw := middleware.NewCircuitBreakerMiddleware(5, 30*time.Second)
//
// Opens after consecutive failures exceed the threshold; recovers after the
// timeout elapses. Stdlib-only — no external dependencies.
//
// # Timeout
//
//	mw := middleware.NewTimeoutMiddleware(60 * time.Second)
//
// Enforces a deadline on both Complete and Stream calls. On timeout,
// Stream returns an error via StreamResult.Err() rather than blocking indefinitely.
package middleware
