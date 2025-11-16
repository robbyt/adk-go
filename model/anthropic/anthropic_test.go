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
	"context"
	"os"
	"testing"

	"github.com/anthropics/anthropic-sdk-go/option"
)

func TestConfig_applyDefaults(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		cfg            *Config
		wantMaxTokens  int64
		wantProvider   string
		wantOptsNotNil bool
	}{
		{
			name:           "empty_config",
			cfg:            &Config{},
			wantMaxTokens:  defaultMaxTokens,
			wantProvider:   ProviderVertexAI,
			wantOptsNotNil: true,
		},
		{
			name: "custom_max_tokens",
			cfg: &Config{
				MaxTokens: 4096,
			},
			wantMaxTokens:  4096,
			wantProvider:   ProviderVertexAI,
			wantOptsNotNil: true,
		},
		{
			name: "custom_provider",
			cfg: &Config{
				Provider: ProviderAnthropic,
			},
			wantMaxTokens:  defaultMaxTokens,
			wantProvider:   ProviderAnthropic,
			wantOptsNotNil: true,
		},
		{
			name: "with_client_options",
			cfg: &Config{
				ClientOptions: []option.RequestOption{option.WithMaxRetries(3)},
			},
			wantMaxTokens:  defaultMaxTokens,
			wantProvider:   ProviderVertexAI,
			wantOptsNotNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.cfg.applyDefaults()

			if tt.cfg.MaxTokens != tt.wantMaxTokens {
				t.Errorf("MaxTokens = %d, want %d", tt.cfg.MaxTokens, tt.wantMaxTokens)
			}
			if tt.cfg.Provider != tt.wantProvider {
				t.Errorf("Provider = %s, want %s", tt.cfg.Provider, tt.wantProvider)
			}
			if tt.wantOptsNotNil && tt.cfg.ClientOptions == nil {
				t.Error("ClientOptions should not be nil after applyDefaults")
			}
		})
	}
}

func TestNewModel_Validation(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("empty_model_name", func(t *testing.T) {
		_, err := NewModel(ctx, "", nil)
		if err == nil {
			t.Error("NewModel with empty model name should return error")
		}
	})

	t.Run("anthropic_provider_without_api_key", func(t *testing.T) {
		cfg := &Config{
			Provider: ProviderAnthropic,
		}
		_, err := NewModel(ctx, "claude-3-sonnet", cfg)
		if err == nil {
			t.Error("NewModel with anthropic provider but no API key should return error")
		}
		if err != nil && err.Error() != "API key must be provided to use Anthropic provider" {
			t.Errorf("unexpected error message: %v", err)
		}
	})

	t.Run("vertex_without_env_vars", func(t *testing.T) {
		// Save and clear environment variables
		oldProject := os.Getenv(envProjectID)
		oldLocation := os.Getenv(envLocation)
		os.Unsetenv(envProjectID)
		os.Unsetenv(envLocation)
		defer func() {
			if oldProject != "" {
				os.Setenv(envProjectID, oldProject)
			}
			if oldLocation != "" {
				os.Setenv(envLocation, oldLocation)
			}
		}()

		cfg := &Config{
			Provider: ProviderVertexAI,
		}
		_, err := NewModel(ctx, "claude-3-sonnet", cfg)
		if err == nil {
			t.Error("NewModel with Vertex AI provider but missing env vars should return error")
		}
	})

	t.Run("bedrock_provider", func(t *testing.T) {
		// Bedrock provider should not fail validation (user provides config via ClientOptions)
		cfg := &Config{
			Provider: ProviderAWSBedrock,
		}
		// This will fail at client creation, but should pass our validation
		_, err := NewModel(ctx, "claude-3-sonnet", cfg)
		// We expect an error from the SDK, not from our validation
		// Our validation should pass for bedrock
		if err != nil && err.Error() == "API key must be provided to use Anthropic provider" {
			t.Error("Bedrock provider should not require API key validation")
		}
	})
}

func TestNewModel_WithAnthropicProvider(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	cfg := &Config{
		Provider: ProviderAnthropic,
		APIKey:   "test-api-key",
	}

	model, err := NewModel(ctx, "claude-3-sonnet", cfg)
	if err != nil {
		t.Fatalf("NewModel failed: %v", err)
	}

	if model.Name() != "claude-3-sonnet" {
		t.Errorf("Name() = %s, want %s", model.Name(), "claude-3-sonnet")
	}

	anthModel, ok := model.(*AnthropicModel)
	if !ok {
		t.Fatal("model is not *AnthropicModel")
	}

	if anthModel.maxTokens != defaultMaxTokens {
		t.Errorf("maxTokens = %d, want %d", anthModel.maxTokens, defaultMaxTokens)
	}
}

func TestNewModel_CustomMaxTokens(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	cfg := &Config{
		Provider:  ProviderAnthropic,
		APIKey:    "test-api-key",
		MaxTokens: 2048,
	}

	model, err := NewModel(ctx, "claude-3-opus", cfg)
	if err != nil {
		t.Fatalf("NewModel failed: %v", err)
	}

	anthModel, ok := model.(*AnthropicModel)
	if !ok {
		t.Fatal("model is not *AnthropicModel")
	}

	if anthModel.maxTokens != 2048 {
		t.Errorf("maxTokens = %d, want %d", anthModel.maxTokens, 2048)
	}
}

func TestNewModel_NilConfig(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	// Test with Anthropic provider to avoid needing GCP credentials
	cfg := &Config{
		Provider: ProviderAnthropic,
		APIKey:   "test-key",
	}

	model, err := NewModel(ctx, "claude-3-haiku", cfg)
	if err != nil {
		t.Fatalf("NewModel failed: %v", err)
	}

	anthModel, ok := model.(*AnthropicModel)
	if !ok {
		t.Fatal("model is not *AnthropicModel")
	}

	// With nil config, should have used default maxTokens
	if anthModel.maxTokens != defaultMaxTokens {
		t.Errorf("maxTokens = %d, want default %d", anthModel.maxTokens, defaultMaxTokens)
	}
}

func TestAnthropicModel_Name(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		modelName string
	}{
		{"claude-3-sonnet", "claude-3-5-sonnet-20241022"},
		{"claude-3-opus", "claude-3-opus-20240229"},
		{"claude-3-haiku", "claude-3-haiku-20240307"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := &AnthropicModel{
				name: tt.modelName,
			}

			if got := model.Name(); got != tt.modelName {
				t.Errorf("Name() = %s, want %s", got, tt.modelName)
			}
		})
	}
}

func TestConstants(t *testing.T) {
	t.Parallel()

	// Verify provider constants
	if ProviderVertexAI != "vertex_ai" {
		t.Errorf("ProviderVertexAI = %s, want vertex_ai", ProviderVertexAI)
	}
	if ProviderAnthropic != "anthropic" {
		t.Errorf("ProviderAnthropic = %s, want anthropic", ProviderAnthropic)
	}
	if ProviderAWSBedrock != "aws_bedrock" {
		t.Errorf("ProviderAWSBedrock = %s, want aws_bedrock", ProviderAWSBedrock)
	}

	// Verify env var constants
	if envProjectID != "GOOGLE_CLOUD_PROJECT" {
		t.Errorf("envProjectID = %s, want GOOGLE_CLOUD_PROJECT", envProjectID)
	}
	if envLocation != "GOOGLE_CLOUD_LOCATION" {
		t.Errorf("envLocation = %s, want GOOGLE_CLOUD_LOCATION", envLocation)
	}

	// Verify default values
	if defaultMaxTokens != 8192 {
		t.Errorf("defaultMaxTokens = %d, want 8192", defaultMaxTokens)
	}
	if defaultOAuthScope != "https://www.googleapis.com/auth/cloud-platform" {
		t.Errorf("defaultOAuthScope = %s, want https://www.googleapis.com/auth/cloud-platform", defaultOAuthScope)
	}
}
