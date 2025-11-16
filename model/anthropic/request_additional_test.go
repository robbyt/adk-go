// Copyright 2025 Google LLC
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

package anthropic

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

func TestBuildContentBlockFromPart_Coverage(t *testing.T) {
	t.Parallel()

	builder := &RequestBuilder{modelName: "test-model", maxTokens: 1024}

	t.Run("function_call", func(t *testing.T) {
		part := &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   "call-123",
				Name: "get_weather",
				Args: map[string]any{"city": "San Francisco"},
			},
		}
		block, err := builder.buildContentBlockFromPart(part)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		name := block.GetName()
		if name == nil || *name == "" {
			t.Error("expected Name to be set")
		}
	})

	t.Run("function_response", func(t *testing.T) {
		part := &genai.Part{
			FunctionResponse: &genai.FunctionResponse{
				ID:       "call-123",
				Response: map[string]any{"result": "sunny"},
			},
		}
		_, err := builder.buildContentBlockFromPart(part)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Tool result block created successfully
	})

	t.Run("image_png", func(t *testing.T) {
		part := &genai.Part{
			InlineData: &genai.Blob{
				MIMEType: "image/png",
				Data:     []byte("fake-png-data"),
			},
		}
		_, err := builder.buildContentBlockFromPart(part)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Image block created successfully
	})

	t.Run("image_jpeg_uppercase", func(t *testing.T) {
		part := &genai.Part{
			InlineData: &genai.Blob{
				MIMEType: "IMAGE/JPEG",
				Data:     []byte("fake-jpeg-data"),
			},
		}
		_, err := builder.buildContentBlockFromPart(part)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Image block created successfully for uppercase MIME
	})

	t.Run("executable_code", func(t *testing.T) {
		part := &genai.Part{
			ExecutableCode: &genai.ExecutableCode{
				Language: "python",
				Code:     "print('hello')",
			},
		}
		block, err := builder.buildContentBlockFromPart(part)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		text := block.GetText()
		if text == nil {
			t.Fatal("expected Text block for executable code")
		}
		if !stringContains(*text, "python") || !stringContains(*text, "print('hello')") {
			t.Errorf("expected code text to contain language and code, got: %s", *text)
		}
	})

	t.Run("code_execution_result", func(t *testing.T) {
		part := &genai.Part{
			CodeExecutionResult: &genai.CodeExecutionResult{
				Output: "hello\n",
			},
		}
		block, err := builder.buildContentBlockFromPart(part)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		text := block.GetText()
		if text == nil {
			t.Fatal("expected Text block for code execution result")
		}
		if !stringContains(*text, "hello") {
			t.Errorf("expected result text to contain output, got: %s", *text)
		}
	})

	t.Run("unsupported_part", func(t *testing.T) {
		part := &genai.Part{}
		_, err := builder.buildContentBlockFromPart(part)
		if err == nil {
			t.Error("expected error for unsupported part")
		}
	})

	t.Run("non_image_inline_data", func(t *testing.T) {
		part := &genai.Part{
			InlineData: &genai.Blob{
				MIMEType: "application/pdf",
				Data:     []byte("fake-pdf"),
			},
		}
		_, err := builder.buildContentBlockFromPart(part)
		if err == nil {
			t.Error("expected error for non-image inline data")
		}
	})
}

func TestBuildSystemInstruction_Coverage(t *testing.T) {
	t.Parallel()

	builder := &RequestBuilder{}

	tests := []struct {
		name    string
		content *genai.Content
		want    int
	}{
		{
			name:    "nil_content",
			content: nil,
			want:    0,
		},
		{
			name:    "empty_parts",
			content: &genai.Content{Parts: []*genai.Part{}},
			want:    0,
		},
		{
			name: "single_text_part",
			content: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("You are a helpful assistant"),
				},
			},
			want: 1,
		},
		{
			name: "multiple_text_parts",
			content: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("You are a helpful assistant"),
					genai.NewPartFromText("Be concise"),
				},
			},
			want: 2,
		},
		{
			name: "nil_part",
			content: &genai.Content{
				Parts: []*genai.Part{nil},
			},
			want: 0,
		},
		{
			name: "empty_text_part",
			content: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText(""),
				},
			},
			want: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			blocks := builder.buildSystemInstruction(tt.content)
			if len(blocks) != tt.want {
				t.Errorf("got %d blocks, want %d", len(blocks), tt.want)
			}
		})
	}
}

func TestBuildMessageFromContent_Coverage(t *testing.T) {
	t.Parallel()

	builder := &RequestBuilder{modelName: "test", maxTokens: 1024}

	tests := []struct {
		name     string
		content  *genai.Content
		wantRole anthropic.MessageParamRole
	}{
		{
			name: "user_role",
			content: &genai.Content{
				Role:  genai.RoleUser,
				Parts: []*genai.Part{genai.NewPartFromText("Hello")},
			},
			wantRole: anthropic.MessageParamRoleUser,
		},
		{
			name: "model_role",
			content: &genai.Content{
				Role:  genai.RoleModel,
				Parts: []*genai.Part{genai.NewPartFromText("Hi")},
			},
			wantRole: anthropic.MessageParamRoleAssistant,
		},
		{
			name: "assistant_role",
			content: &genai.Content{
				Role:  "assistant",
				Parts: []*genai.Part{genai.NewPartFromText("Hi")},
			},
			wantRole: anthropic.MessageParamRoleAssistant,
		},
		{
			name: "unknown_role_defaults_to_user",
			content: &genai.Content{
				Role:  "system",
				Parts: []*genai.Part{genai.NewPartFromText("Test")},
			},
			wantRole: anthropic.MessageParamRoleUser,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg, err := builder.buildMessageFromContent(tt.content)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if msg.Role != tt.wantRole {
				t.Errorf("role = %v, want %v", msg.Role, tt.wantRole)
			}
		})
	}
}

func stringContains(s, substr string) bool {
	return len(s) >= len(substr) && stringFind(s, substr)
}

func stringFind(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
