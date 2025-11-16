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

	"github.com/anthropics/anthropic-sdk-go"
)

// AnthropicError wraps errors from the Anthropic API with additional context
// and normalized error information for the ADK framework.
type AnthropicError struct {
	// StatusCode is the HTTP status code from the API response
	StatusCode int
	// Type is the error type returned by Anthropic (e.g., "invalid_request_error")
	Type string
	// Message is the human-readable error message
	Message string
	// RequestID is the Anthropic request ID for debugging
	RequestID string
	// Err is the underlying error
	Err error
}

func (e *AnthropicError) Error() string {
	if e.RequestID != "" {
		return fmt.Sprintf("anthropic api error (status=%d, type=%s, request_id=%s): %s",
			e.StatusCode, e.Type, e.RequestID, e.Message)
	}
	return fmt.Sprintf("anthropic api error (status=%d, type=%s): %s",
		e.StatusCode, e.Type, e.Message)
}

func (e *AnthropicError) Unwrap() error {
	return e.Err
}

// IsRateLimitError returns true if the error is a rate limit error (429).
func (e *AnthropicError) IsRateLimitError() bool {
	return e.StatusCode == http.StatusTooManyRequests
}

// IsAuthenticationError returns true if the error is an authentication error (401/403).
func (e *AnthropicError) IsAuthenticationError() bool {
	return e.StatusCode == http.StatusUnauthorized || e.StatusCode == http.StatusForbidden
}

// IsInvalidRequestError returns true if the error is an invalid request error (400).
func (e *AnthropicError) IsInvalidRequestError() bool {
	return e.StatusCode == http.StatusBadRequest
}

// IsServerError returns true if the error is a server error (5xx).
func (e *AnthropicError) IsServerError() bool {
	return e.StatusCode >= 500 && e.StatusCode < 600
}

// normalizeError converts Anthropic SDK errors into normalized AnthropicError instances
// with extracted metadata. If the error is not from the Anthropic API, it is returned
// as-is.
func normalizeError(err error) error {
	if err == nil {
		return nil
	}

	var apiErr *anthropic.Error
	if !errors.As(err, &apiErr) {
		// Not an Anthropic API error, return as-is
		return err
	}

	// Extract error details from the Anthropic API error
	// Note: Type and Message are not directly accessible as exported fields,
	// so we use the error message and infer type from status code
	errType := inferErrorType(apiErr.StatusCode)
	normalizedErr := &AnthropicError{
		StatusCode: apiErr.StatusCode,
		Type:       errType,
		Message:    err.Error(), // Use the error message from Error() method
		RequestID:  apiErr.RequestID,
		Err:        err,
	}

	return normalizedErr
}

// inferErrorType infers the error type based on HTTP status code
func inferErrorType(statusCode int) string {
	switch statusCode {
	case http.StatusBadRequest:
		return "invalid_request_error"
	case http.StatusUnauthorized, http.StatusForbidden:
		return "authentication_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusInternalServerError, http.StatusBadGateway, http.StatusServiceUnavailable:
		return "api_error"
	default:
		return "api_error"
	}
}

// getErrorCode returns a standardized error code string based on the error type.
// This can be used to populate the ErrorCode field in model.LLMResponse.
func getErrorCode(err error) string {
	var anthErr *AnthropicError
	if errors.As(err, &anthErr) {
		switch {
		case anthErr.IsRateLimitError():
			return "RATE_LIMIT_EXCEEDED"
		case anthErr.IsAuthenticationError():
			return "AUTHENTICATION_FAILED"
		case anthErr.IsInvalidRequestError():
			return "INVALID_REQUEST"
		case anthErr.IsServerError():
			return "SERVER_ERROR"
		default:
			return "API_ERROR"
		}
	}

	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) {
		// Fallback for non-normalized errors
		if apiErr.StatusCode == http.StatusTooManyRequests {
			return "RATE_LIMIT_EXCEEDED"
		}
		return "API_ERROR"
	}

	return "UNKNOWN_ERROR"
}
