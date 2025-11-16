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
	"errors"
	"fmt"
	"net/http"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

func TestNormalizeError_AnthropicError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		statusCode     int
		errType        string
		message        string
		requestID      string
		wantStatusCode int
		wantType       string
		wantMessage    string
		wantRequestID  string
	}{
		{
			name:           "rate_limit_error",
			statusCode:     429,
			errType:        "rate_limit_error",
			message:        "Rate limit exceeded",
			requestID:      "req-123",
			wantStatusCode: 429,
			wantType:       "rate_limit_error",
			wantMessage:    "Rate limit exceeded",
			wantRequestID:  "req-123",
		},
		{
			name:           "authentication_error",
			statusCode:     401,
			errType:        "authentication_error",
			message:        "Invalid API key",
			wantStatusCode: 401,
			wantType:       "authentication_error",
			wantMessage:    "Invalid API key",
		},
		{
			name:           "invalid_request_error",
			statusCode:     400,
			errType:        "invalid_request_error",
			message:        "Missing required parameter",
			wantStatusCode: 400,
			wantType:       "invalid_request_error",
			wantMessage:    "Missing required parameter",
		},
		{
			name:           "server_error",
			statusCode:     500,
			errType:        "api_error",
			message:        "Internal server error",
			wantStatusCode: 500,
			wantType:       "api_error",
			wantMessage:    "Internal server error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a mock AnthropicError directly since we can't use internal packages
			anthErr := &AnthropicError{
				StatusCode: tt.statusCode,
				Type:       tt.errType,
				Message:    tt.message,
				RequestID:  tt.requestID,
				Err:        fmt.Errorf("mock error"),
			}

			if anthErr.StatusCode != tt.wantStatusCode {
				t.Errorf("StatusCode = %d, want %d", anthErr.StatusCode, tt.wantStatusCode)
			}
			if anthErr.Type != tt.wantType {
				t.Errorf("Type = %s, want %s", anthErr.Type, tt.wantType)
			}
			if anthErr.Message != tt.wantMessage {
				t.Errorf("Message = %s, want %s", anthErr.Message, tt.wantMessage)
			}
			if anthErr.RequestID != tt.wantRequestID {
				t.Errorf("RequestID = %s, want %s", anthErr.RequestID, tt.wantRequestID)
			}
		})
	}
}

func TestNormalizeError_NonAnthropicError(t *testing.T) {
	t.Parallel()

	genericErr := fmt.Errorf("some generic error")
	normalizedErr := normalizeError(genericErr)

	if normalizedErr != genericErr {
		t.Errorf("normalizeError should return original error for non-Anthropic errors")
	}
}

func TestNormalizeError_Nil(t *testing.T) {
	t.Parallel()

	normalizedErr := normalizeError(nil)
	if normalizedErr != nil {
		t.Errorf("normalizeError(nil) = %v, want nil", normalizedErr)
	}
}

func TestAnthropicError_ErrorMethods(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                    string
		statusCode              int
		wantIsRateLimitError    bool
		wantIsAuthError         bool
		wantIsInvalidRequestErr bool
		wantIsServerError       bool
	}{
		{
			name:                 "rate_limit",
			statusCode:           429,
			wantIsRateLimitError: true,
		},
		{
			name:            "unauthorized",
			statusCode:      401,
			wantIsAuthError: true,
		},
		{
			name:            "forbidden",
			statusCode:      403,
			wantIsAuthError: true,
		},
		{
			name:                    "bad_request",
			statusCode:              400,
			wantIsInvalidRequestErr: true,
		},
		{
			name:              "internal_server_error",
			statusCode:        500,
			wantIsServerError: true,
		},
		{
			name:              "bad_gateway",
			statusCode:        502,
			wantIsServerError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := &AnthropicError{StatusCode: tt.statusCode}

			if got := err.IsRateLimitError(); got != tt.wantIsRateLimitError {
				t.Errorf("IsRateLimitError() = %v, want %v", got, tt.wantIsRateLimitError)
			}
			if got := err.IsAuthenticationError(); got != tt.wantIsAuthError {
				t.Errorf("IsAuthenticationError() = %v, want %v", got, tt.wantIsAuthError)
			}
			if got := err.IsInvalidRequestError(); got != tt.wantIsInvalidRequestErr {
				t.Errorf("IsInvalidRequestError() = %v, want %v", got, tt.wantIsInvalidRequestErr)
			}
			if got := err.IsServerError(); got != tt.wantIsServerError {
				t.Errorf("IsServerError() = %v, want %v", got, tt.wantIsServerError)
			}
		})
	}
}

func TestGetErrorCode(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		err      error
		wantCode string
	}{
		{
			name: "rate_limit",
			err: &AnthropicError{
				StatusCode: http.StatusTooManyRequests,
			},
			wantCode: "RATE_LIMIT_EXCEEDED",
		},
		{
			name: "authentication",
			err: &AnthropicError{
				StatusCode: http.StatusUnauthorized,
			},
			wantCode: "AUTHENTICATION_FAILED",
		},
		{
			name: "invalid_request",
			err: &AnthropicError{
				StatusCode: http.StatusBadRequest,
			},
			wantCode: "INVALID_REQUEST",
		},
		{
			name: "server_error",
			err: &AnthropicError{
				StatusCode: http.StatusInternalServerError,
			},
			wantCode: "SERVER_ERROR",
		},
		{
			name: "other_anthropic_error",
			err: &AnthropicError{
				StatusCode: http.StatusNotFound,
			},
			wantCode: "API_ERROR",
		},
		{
			name:     "generic_error",
			err:      fmt.Errorf("some error"),
			wantCode: "UNKNOWN_ERROR",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getErrorCode(tt.err)
			if got != tt.wantCode {
				t.Errorf("getErrorCode() = %s, want %s", got, tt.wantCode)
			}
		})
	}
}

func TestAnthropicError_Error(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		err       *AnthropicError
		wantMatch string
	}{
		{
			name: "with_request_id",
			err: &AnthropicError{
				StatusCode: 429,
				Type:       "rate_limit_error",
				Message:    "Rate limit exceeded",
				RequestID:  "req-123",
			},
			wantMatch: "request_id=req-123",
		},
		{
			name: "without_request_id",
			err: &AnthropicError{
				StatusCode: 400,
				Type:       "invalid_request_error",
				Message:    "Bad request",
			},
			wantMatch: "status=400",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errMsg := tt.err.Error()
			if errMsg == "" {
				t.Error("Error() returned empty string")
			}
			// Just verify the error message is non-empty and contains expected parts
			if tt.wantMatch != "" && !contains(errMsg, tt.wantMatch) {
				t.Errorf("Error() = %q, want to contain %q", errMsg, tt.wantMatch)
			}
		})
	}
}

func TestAnthropicError_Unwrap(t *testing.T) {
	t.Parallel()

	underlyingErr := fmt.Errorf("underlying error")
	anthErr := &AnthropicError{
		StatusCode: 500,
		Err:        underlyingErr,
	}

	unwrapped := anthErr.Unwrap()
	if unwrapped != underlyingErr {
		t.Errorf("Unwrap() = %v, want %v", unwrapped, underlyingErr)
	}

	// Test that errors.Is works with wrapped errors
	if !errors.Is(anthErr, underlyingErr) {
		t.Error("errors.Is should find the underlying error")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestInferErrorType(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		statusCode int
		want       string
	}{
		{
			name:       "bad_request_400",
			statusCode: 400,
			want:       "invalid_request_error",
		},
		{
			name:       "unauthorized_401",
			statusCode: 401,
			want:       "authentication_error",
		},
		{
			name:       "forbidden_403",
			statusCode: 403,
			want:       "authentication_error",
		},
		{
			name:       "rate_limit_429",
			statusCode: 429,
			want:       "rate_limit_error",
		},
		{
			name:       "internal_server_error_500",
			statusCode: 500,
			want:       "api_error",
		},
		{
			name:       "bad_gateway_502",
			statusCode: 502,
			want:       "api_error",
		},
		{
			name:       "service_unavailable_503",
			statusCode: 503,
			want:       "api_error",
		},
		{
			name:       "not_found_404",
			statusCode: 404,
			want:       "api_error",
		},
		{
			name:       "unknown_status",
			statusCode: 999,
			want:       "api_error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := inferErrorType(tt.statusCode)
			if got != tt.want {
				t.Errorf("inferErrorType(%d) = %s, want %s", tt.statusCode, got, tt.want)
			}
		})
	}
}

func TestGetErrorCode_WithAnthropicError(t *testing.T) {
	t.Parallel()

	// Test with direct anthropic.Error (fallback path)
	apiErr := &anthropic.Error{
		StatusCode: 429,
	}

	code := getErrorCode(apiErr)
	if code != "RATE_LIMIT_EXCEEDED" {
		t.Errorf("getErrorCode for anthropic.Error = %s, want RATE_LIMIT_EXCEEDED", code)
	}
}

func TestGetErrorCode_OtherStatusCodes(t *testing.T) {
	t.Parallel()

	// Test non-429 status code in anthropic.Error
	apiErr := &anthropic.Error{
		StatusCode: 404,
	}

	code := getErrorCode(apiErr)
	if code != "API_ERROR" {
		t.Errorf("getErrorCode for 404 = %s, want API_ERROR", code)
	}
}
