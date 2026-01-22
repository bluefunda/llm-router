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
//	import "github.com/bluefunda/llm-router/middleware"
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
