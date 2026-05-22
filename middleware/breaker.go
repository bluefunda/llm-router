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

package middleware

import (
	"sync"
	"time"
)

// CBState is the state of the circuit breaker.
type CBState int

const (
	CBStateClosed   CBState = iota // requests pass through normally
	CBStateOpen                    // requests rejected immediately
	CBStateHalfOpen                // limited probe requests allowed to test recovery
)

func (s CBState) String() string {
	switch s {
	case CBStateClosed:
		return "closed"
	case CBStateOpen:
		return "open"
	case CBStateHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

// circuitBreaker is a stdlib-only three-state circuit breaker.
//
// State machine:
//   - Closed: all requests pass through. Trip to Open after consecutive
//     failures exceed maxFailures within a resetInterval.
//   - Open: all requests rejected. After openTimeout, transition to HalfOpen.
//   - HalfOpen: allow up to maxFailures probe requests. All succeed → Closed.
//     Any fail → Open.
type circuitBreaker struct {
	mu sync.Mutex

	maxFailures   uint32
	openTimeout   time.Duration
	resetInterval time.Duration

	state               CBState
	consecutiveFailures uint32
	openUntil           time.Time
	lastIntervalReset   time.Time
	halfOpenRemaining   uint32
}

func newCircuitBreaker(maxFailures uint32, openTimeout time.Duration) *circuitBreaker {
	return &circuitBreaker{
		maxFailures:       maxFailures,
		openTimeout:       openTimeout,
		resetInterval:     60 * time.Second,
		state:             CBStateClosed,
		lastIntervalReset: time.Now(),
		halfOpenRemaining: maxFailures,
	}
}

// State returns the current circuit breaker state.
func (cb *circuitBreaker) State() CBState {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.effectiveState()
}

// effectiveState transitions Open→HalfOpen when the timeout has elapsed.
// Must be called with cb.mu held.
func (cb *circuitBreaker) effectiveState() CBState {
	if cb.state == CBStateOpen && time.Now().After(cb.openUntil) {
		cb.state = CBStateHalfOpen
		cb.halfOpenRemaining = cb.maxFailures
	}
	return cb.state
}

// Allow reports whether the circuit breaker will permit a request.
// Returns false when the breaker is open or half-open probes are exhausted.
// Must be paired with a call to Record after the request completes.
func (cb *circuitBreaker) Allow() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.effectiveState() {
	case CBStateOpen:
		return false
	case CBStateHalfOpen:
		if cb.halfOpenRemaining == 0 {
			return false
		}
		cb.halfOpenRemaining--
	case CBStateClosed:
		if time.Now().After(cb.lastIntervalReset.Add(cb.resetInterval)) {
			cb.consecutiveFailures = 0
			cb.lastIntervalReset = time.Now()
		}
	}
	return true
}

// Record updates circuit breaker state after a request completes.
// Pass nil on success, non-nil on failure.
func (cb *circuitBreaker) Record(err error) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if err != nil {
		switch cb.state {
		case CBStateClosed:
			cb.consecutiveFailures++
			if cb.consecutiveFailures > cb.maxFailures {
				cb.state = CBStateOpen
				cb.openUntil = time.Now().Add(cb.openTimeout)
			}
		case CBStateHalfOpen:
			cb.state = CBStateOpen
			cb.openUntil = time.Now().Add(cb.openTimeout)
		}
		return
	}

	switch cb.state {
	case CBStateClosed:
		cb.consecutiveFailures = 0
	case CBStateHalfOpen:
		cb.state = CBStateClosed
		cb.consecutiveFailures = 0
	}
}
