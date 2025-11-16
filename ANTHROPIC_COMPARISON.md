# Comparison: ADK-Go vs Fantasy Anthropic SDK Usage

This document compares how the ADK-Go integration (PR #233) and the Charm Fantasy library use the official Anthropic SDK.

## Overview

Both implementations provide abstractions over the Anthropic SDK, but with different architectural patterns and feature sets.

- **ADK-Go**: Implements `model.LLM` interface, targeting Vertex AI as primary deployment
- **Fantasy**: Implements `fantasy.Provider` interface with multi-deployment focus

## Key Differences

### 1. Client Initialization Timing

**ADK-Go (this PR)**:
```go
// Client created at NewModel time and stored
func NewModel(ctx context.Context, modelName string, cfg *Config) (model.LLM, error) {
    // ... build opts ...
    return &AnthropicModel{
        name:      modelName,
        maxTokens: cfg.MaxTokens,
        client:    anthropic.NewClient(opts...),  // Created here, stored
    }, nil
}
```

**Fantasy**:
```go
// Client created lazily per-call in LanguageModel method
func (a *provider) LanguageModel(ctx context.Context, call fantasy.Call) iter.Seq2[fantasy.StreamPart, error] {
    clientOptions := buildClientOptions(ctx, a.options)  // Built on-demand
    a.client = anthropic.NewClient(clientOptions...)
    // ...
}
```

**Implication**:
- ADK-Go: Single client instance, faster subsequent calls, but less flexible for dynamic configuration
- Fantasy: Fresh client per call, more overhead but supports runtime configuration changes

---

### 2. Vertex AI Authentication

**ADK-Go**:
```go
opts = append(opts, vertex.WithGoogleAuth(ctx, location, projectID, defaultOAuthScope))
```
- Simpler API
- Uses OAuth scope: `https://www.googleapis.com/auth/cloud-platform`
- No credential customization support

**Fantasy**:
```go
var credentials *google.Credentials
if a.options.skipAuth {
    credentials = &google.Credentials{TokenSource: &googleDummyTokenSource{}}
} else {
    credentials, err = google.FindDefaultCredentials(ctx)
}
opts = append(opts, vertex.WithCredentials(ctx, location, project, credentials))
```
- More explicit credential handling
- Supports `skipAuth` for testing/special scenarios
- Uses `google.FindDefaultCredentials()` for ADC

**Implication**: Fantasy has more flexibility for testing and custom auth scenarios

---

### 3. Provider Configuration Pattern

**ADK-Go**:
```go
type Config struct {
    Provider      string  // "vertex_ai", "anthropic", "aws_bedrock"
    APIKey        string
    MaxTokens     int64
    ClientOptions []option.RequestOption
}

// Switch-based selection
switch cfg.Provider {
case ProviderAnthropic:
    opts = append(opts, option.WithAPIKey(cfg.APIKey))
case ProviderAWSBedrock:
    // User provides bedrock.WithConfig()
default:
    opts = append(opts, vertex.WithGoogleAuth(...))
}
```

**Fantasy**:
```go
// Functional options pattern
func WithVertex(project, location string) Option
func WithAPIKey(apiKey string) Option
func WithBedrock(region string, opts ...bedrock.Option) Option

// Usage:
New(
    WithVertex("my-project", "us-central1"),
    WithMaxTokens(4096),
)
```

**Implication**:
- ADK-Go: More compact but less discoverable
- Fantasy: More Go-idiomatic with better type safety and discoverability

---

### 4. Stream Event Processing

**ADK-Go**:
```go
func parsePartialStreamEvent(event anthropic.MessageStreamEventUnion) *model.LLMResponse {
    deltaEvent, ok := event.AsAny().(anthropic.ContentBlockDeltaEvent)
    if !ok {
        return nil  // Ignores non-delta events
    }

    switch v := deltaEvent.Delta.AsAny().(type) {
    case anthropic.TextDelta:
        part = genai.NewPartFromText(v.Text)
    case anthropic.ThinkingDelta:
        part = &genai.Part{Text: v.Thinking, Thought: true}
    default:
        return nil
    }
}
```
- Only yields `ContentBlockDeltaEvent` types
- Handles: `TextDelta`, `ThinkingDelta`
- Ignores: `content_block_start`, `content_block_stop`, `input_json_delta`, `signature_delta`

**Fantasy**:
```go
// Processes multiple event types
switch chunk := stream.Current().AsAny().(type) {
case anthropic.ContentBlockStartEvent:
    // Yields block start with IDs
case anthropic.ContentBlockDeltaEvent:
    // Handles TextDelta, ThinkingDelta, InputJSONDelta
case anthropic.MessageDeltaEvent:
    // Handles SignatureDelta for reasoning metadata
case anthropic.ContentBlockStopEvent:
    // Yields block completion
}
```

**Implication**: Fantasy provides more granular streaming control and metadata

---

### 5. Extended Thinking / Reasoning Support

**ADK-Go**:
```go
case anthropic.ThinkingDelta:
    if v.Thinking != "" {
        part = &genai.Part{Text: v.Thinking, Thought: true}
    }
```
- Basic support: captures thinking deltas with `Thought: true` flag
- No configuration for thinking budget/tokens
- No signature/metadata preservation

**Fantasy**:
```go
if call.Extended.Enabled {
    if call.Extended.BudgetTokens > 0 {
        params.Extended.Thinking = &extended.Thinking{
            Type:         extended.ThinkingTypeEnabled,
            BudgetTokens: ptr(int64(call.Extended.BudgetTokens)),
        }
    }
}

// Also handles signature deltas:
case anthropic.MessageDeltaEvent:
    if delta.SignatureDelta != nil {
        // Preserve signing metadata
    }
```

**Implication**: Fantasy has first-class extended thinking support with budget control

---

### 6. Prompt Caching

**ADK-Go**:
- No prompt caching implementation visible
- All content treated equally

**Fantasy**:
```go
// Applies cache control to appropriate message boundaries
if shouldCache {
    block.CacheControl = &anthropic.CacheControlEphemeralParam{
        Type: constant.ValueOf[constant.Ephemeral](),
    }
}
```

**Implication**: Fantasy can reduce costs via prompt caching for repeated contexts

---

### 7. Message Grouping/Consolidation

**ADK-Go**:
```go
// Direct 1:1 conversion
for _, content := range req.Contents {
    message, err := builder.buildMessageFromContent(content)
    messages = append(messages, message)
}
```
- Each `genai.Content` becomes one `MessageParam`
- No role consolidation

**Fantasy**:
```go
// Groups consecutive same-role messages
blocks := groupIntoBlocks(call.Messages)
for _, block := range blocks {
    // Consolidates multiple messages with same role
}
```

**Implication**: Fantasy handles multi-turn conversations more efficiently

---

### 8. Tool Result Serialization

**ADK-Go**:
```go
func stringifyFunctionResponse(resp map[string]any) string {
    if result, ok := resp["result"]; ok && result != nil {
        return fmt.Sprint(result)
    }
    if output, ok := resp["output"]; ok && output != nil {
        return fmt.Sprint(output)
    }
    return string(json.Marshal(resp))  // Fallback to JSON string
}

// Creates text-based tool result
anthropic.NewToolResultBlock(funcResponse.ID, content, false)
```
- Converts all tool results to strings
- Loses type information
- Simple but potentially lossy

**Fantasy**:
- Likely preserves structured data in tool results
- More faithful to Anthropic's API expectations

**Implication**: ADK-Go may lose precision in complex tool results

---

### 9. Schema Normalization

**ADK-Go**:
```go
func schemaToMap(schema *genai.Schema) (map[string]any, error) {
    raw, err := json.Marshal(schema)
    var result map[string]any
    json.Unmarshal(raw, &result)
    normalizeTypeStrings(result)  // Recursively lowercase all "type" fields
    return result, nil
}
```
- Marshal/unmarshal approach to convert structs to maps
- Recursive normalization of type strings to lowercase
- Handles nested schemas, anyOf, properties, items

**Both implementations** handle the challenge that `genai.Schema.Type` may be uppercase (e.g., `"STRING"`) while Anthropic expects lowercase (e.g., `"string"`).

---

### 10. Error Handling

**ADK-Go** (Updated):
```go
// Disable retries by default for predictable error handling
opts = append(opts, option.WithMaxRetries(0))

// Normalize errors
if err != nil {
    return nil, normalizeError(err)
}

// AnthropicError wrapper with status code, type, message, request ID
type AnthropicError struct {
    StatusCode int
    Type       string
    Message    string
    RequestID  string
    Err        error
}

// Helper methods: IsRateLimitError(), IsAuthenticationError(), etc.
```
- ✅ Error normalization with structured AnthropicError type
- ✅ Explicit retry configuration disabled (option.WithMaxRetries(0))
- ✅ Helper methods for error type identification
- ✅ Preserves request IDs for debugging
- ✅ Comprehensive test coverage

**Fantasy**:
```go
clientOptions = append(clientOptions, option.WithMaxRetries(0))

func toProviderErr(err error) error {
    // Normalizes Anthropic errors to fantasy error types
    var apiErr *anthropic.Error
    if errors.As(err, &apiErr) {
        // Maps to fantasy.ErrRateLimited, etc.
    }
}
```

**Implication**: Both now have sophisticated error handling with explicit no-retry policy and error normalization

---

### 11. Code Execution Handling

**ADK-Go**:
```go
case part.ExecutableCode != nil:
    code := fmt.Sprintf("Code:```%s\n%s\n```",
        part.ExecutableCode.Language, part.ExecutableCode.Code)
    return anthropic.NewTextBlock(code), nil

case part.CodeExecutionResult != nil:
    output := fmt.Sprintf("Execution Result:```code_output\n%s\n```",
        part.CodeExecutionResult.Output)
    return anthropic.NewTextBlock(output), nil
```
- Converts code execution to markdown-formatted text blocks
- Embeds results as text
- No native code execution support from Anthropic

**Fantasy**:
- Likely doesn't handle code execution (not an Anthropic feature)

**Implication**: ADK-Go has special handling for ADK's code execution feature

---

### 12. Default Model Behavior

**ADK-Go**:
```go
const defaultMaxTokens = 8192
```
- Fixed default
- Can be overridden via `Config.MaxTokens` or `GenerateContentConfig.MaxOutputTokens`

**Fantasy**:
- Likely uses Anthropic SDK defaults (varies by model)
- More flexible per-call configuration

---

## Summary Matrix

| Feature | ADK-Go (PR #233) | Fantasy |
|---------|------------------|---------|
| **Client Lifecycle** | Created once, reused | Created per-call |
| **Primary Deployment** | Vertex AI | Multi-provider |
| **Vertex Auth** | `WithGoogleAuth()` | `WithCredentials()` + ADC |
| **Config Pattern** | Struct + switch | Functional options |
| **Extended Thinking** | Basic delta capture | Full budget control |
| **Prompt Caching** | ❌ Not implemented | ✅ Implemented |
| **Message Grouping** | Direct 1:1 | Role consolidation |
| **Stream Events** | Delta events only | All event types |
| **Tool Results** | Stringified | Structured |
| **Error Handling** | ✅ Normalized + no retries | ✅ Normalized + no retries |
| **Retry Policy** | ✅ Explicit disabled | ✅ Explicit disabled |
| **Code Execution** | ✅ Special handling | ❌ N/A |
| **Testing Support** | Standard | `skipAuth` option |

---

## Recommendations for ADK-Go

Based on Fantasy's implementation, consider these potential improvements:

1. **Add prompt caching support** - Can significantly reduce costs for repeated contexts
2. ~~**Implement explicit retry configuration**~~ - ✅ **DONE** - `option.WithMaxRetries(0)` for predictability
3. **Consider lazy client initialization** - For better testability and dynamic config
4. **Enhance stream event handling** - Expose `content_block_start/stop` for richer UX
5. ~~**Add error normalization**~~ - ✅ **DONE** - Map Anthropic errors to ADK error types
6. **Support extended thinking budgets** - Allow configuration of thinking token limits
7. **Improve tool result handling** - Preserve structure instead of stringifying
8. **Use functional options** - More Go-idiomatic than struct-based config

### Completed Improvements

- ✅ **Error Normalization** (implemented in commit 925736c)
  - Added `AnthropicError` wrapper with status code, type, message, and request ID
  - Helper methods for error type identification (rate limit, auth, invalid request, server errors)
  - Comprehensive test coverage
  - All errors normalized through `normalizeError()` function

- ✅ **Retry Policy** (implemented in commit 925736c)
  - Explicit `option.WithMaxRetries(0)` for predictable behavior
  - Users can override via `ClientOptions` if needed

## Notes

- Both use the same official SDK: `github.com/anthropics/anthropic-sdk-go`
- Both leverage the SDK's `message.Accumulate()` pattern correctly
- Both support the same basic streaming workflow
- Fantasy is more feature-complete and production-hardened
- ADK-Go has unique code execution support for the ADK framework
