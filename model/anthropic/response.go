package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// ThinkingBlock represents Claude's internal reasoning/thinking content.
// This content is preserved for context but not included in the main response
// sent to clients.
type ThinkingBlock struct {
	// Thinking contains the raw thinking content
	Thinking string `json:"thinking"`
	// Signature contains cryptographic signature for thinking verification
	Signature string `json:"signature,omitempty"`
	// Type is "thinking" or "redacted_thinking"
	Type string `json:"type"`
}

// ThinkingContext stores all thinking blocks for a response.
// This is used to preserve thinking content for future requests without
// exposing it to the client.
type ThinkingContext struct {
	Blocks []ThinkingBlock `json:"blocks"`
}

func parsePartialStreamEvent(event anthropic.MessageStreamEventUnion) *model.LLMResponse {
	deltaEvent, ok := event.AsAny().(anthropic.ContentBlockDeltaEvent)
	if !ok {
		return nil
	}

	var part *genai.Part
	var thinkingDelta string

	switch v := deltaEvent.Delta.AsAny().(type) {
	case anthropic.TextDelta:
		if v.Text != "" {
			part = genai.NewPartFromText(v.Text)
		}
	case anthropic.ThinkingDelta:
		// Store thinking delta separately, don't include in content
		if v.Thinking != "" {
			thinkingDelta = v.Thinking
		}
	default:
		// Unsupported delta type for partial response
		return nil
	}

	response := &model.LLMResponse{
		CustomMetadata: make(map[string]any),
	}

	// Only include non-thinking content in the main response
	if part != nil {
		content := genai.NewContentFromParts([]*genai.Part{part}, genai.RoleModel)
		response.Content = content
	}

	// Store thinking delta in metadata for accumulation
	if thinkingDelta != "" {
		response.CustomMetadata["thinking_delta"] = thinkingDelta
	}

	// Return nil if we have no content to send to client
	if part == nil && thinkingDelta == "" {
		return nil
	}

	return response
}

type ResponseBuilder struct{}

func (builder *ResponseBuilder) FromMessage(message *anthropic.Message) (*model.LLMResponse, error) {
	var parts []*genai.Part
	var thinkingBlocks []ThinkingBlock

	for _, block := range message.Content {
		part, thinkingBlock, err := builder.buildPartFromContentBlock(block)
		if err != nil {
			return nil, err
		}

		// Separate thinking blocks from regular content
		if thinkingBlock != nil {
			thinkingBlocks = append(thinkingBlocks, *thinkingBlock)
		} else if part != nil {
			parts = append(parts, part)
		}
	}

	content := genai.NewContentFromParts(parts, genai.RoleModel)

	llmResponse := &model.LLMResponse{
		Content:        content,
		FinishReason:   builder.buildFinishReason(message.StopReason),
		UsageMetadata:  builder.extractUsage(message.Usage),
		CustomMetadata: make(map[string]any),
	}

	// Store thinking blocks in metadata for context preservation
	if len(thinkingBlocks) > 0 {
		llmResponse.CustomMetadata["thinking_context"] = ThinkingContext{
			Blocks: thinkingBlocks,
		}
	}

	if message.StopReason != "" {
		llmResponse.CustomMetadata["stop_reason"] = message.StopReason
	}
	if message.StopSequence != "" {
		llmResponse.CustomMetadata["stop_sequence"] = message.StopSequence
	}
	return llmResponse, nil
}

// buildPartFromContentBlock converts an Anthropic content block to a genai.Part.
// For thinking blocks, it returns the thinking content separately.
// Returns (part, thinkingBlock, error)
func (builder *ResponseBuilder) buildPartFromContentBlock(block anthropic.ContentBlockUnion) (*genai.Part, *ThinkingBlock, error) {
	blockType := strings.ToLower(block.Type)

	switch {
	case blockType == string(constant.ValueOf[constant.Text]()):
		return genai.NewPartFromText(block.Text), nil, nil

	case blockType == string(constant.ValueOf[constant.Thinking]()):
		// Don't include thinking in regular content, store separately
		thinking := &ThinkingBlock{
			Thinking:  block.Thinking,
			Signature: block.Signature,
			Type:      string(constant.ValueOf[constant.Thinking]()),
		}
		return nil, thinking, nil

	case blockType == string(constant.ValueOf[constant.RedactedThinking]()):
		// Handle redacted thinking (no actual content, just metadata)
		thinking := &ThinkingBlock{
			Thinking: "[REDACTED]",
			Type:     string(constant.ValueOf[constant.RedactedThinking]()),
		}
		return nil, thinking, nil

	case blockType == string(constant.ValueOf[constant.ToolUse]()):
		args := make(map[string]any)
		if len(block.Input) > 0 {
			if err := json.Unmarshal(block.Input, &args); err != nil {
				return nil, nil, fmt.Errorf("failed to decode tool input: %w", err)
			}
		}
		part := &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   block.ID,
				Name: block.Name,
				Args: args,
			},
		}
		return part, nil, nil

	default:
		return nil, nil, fmt.Errorf("not supported block type '%s' yet", blockType)
	}
}

func (builder *ResponseBuilder) extractUsage(usage anthropic.Usage) *genai.GenerateContentResponseUsageMetadata {
	total := usage.InputTokens + usage.OutputTokens
	return &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:        int32(usage.InputTokens),
		CandidatesTokenCount:    int32(usage.OutputTokens),
		TotalTokenCount:         int32(total),
		CachedContentTokenCount: int32(usage.CacheReadInputTokens),
	}
}

func (builder *ResponseBuilder) buildFinishReason(stop anthropic.StopReason) genai.FinishReason {
	switch stop {
	case anthropic.StopReasonEndTurn, anthropic.StopReasonStopSequence, anthropic.StopReasonToolUse:
		return genai.FinishReasonStop
	case anthropic.StopReasonMaxTokens:
		return genai.FinishReasonMaxTokens
	default:
		return genai.FinishReasonUnspecified
	}
}
