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

package llmrouter

// Option configures the Router
type Option func(*Router)

// WithProvider registers a provider with the router
func WithProvider(name string, p Provider) Option {
	return func(r *Router) {
		r.providers[name] = p
	}
}

// WithModelMapping maps a model to a specific provider
func WithModelMapping(model, provider string) Option {
	return func(r *Router) {
		r.modelMap[model] = provider
	}
}

// WithFallback sets fallback providers in priority order
func WithFallback(providers ...string) Option {
	return func(r *Router) {
		r.fallbacks = providers
	}
}

// WithMiddleware adds middleware to the processing chain.
// Use this with middleware from the middleware package:
//
//	import "github.com/bluefunda/llmrouter/middleware"
//
//	router := llmrouter.New(
//	    llmrouter.WithMiddleware(
//	        middleware.NewRetryMiddleware(3, time.Second),
//	        middleware.NewTimeoutMiddleware(60*time.Second),
//	    ),
//	)
func WithMiddleware(m ...Middleware) Option {
	return func(r *Router) {
		r.middleware = append(r.middleware, m...)
	}
}
