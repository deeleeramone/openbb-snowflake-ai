//! SSE Event output module for CLI.
//!
//! Provides event emitters matching the `openbb_ai` Python package format,
//! supporting both JSON mode (machine-parseable) and human-readable interactive mode.

use serde::{Deserialize, Serialize};
use std::io::{self, Write};

// ============================================================================
// Event Types
// ============================================================================

/// Output mode for events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// JSON mode - single-line JSON per event to stdout
    Json,
    /// Interactive mode - colored human-readable output
    Interactive,
}

impl Default for OutputMode {
    fn default() -> Self {
        OutputMode::Interactive
    }
}

/// SSE event types matching openbb_ai format
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventType {
    Message,
    Reasoning,
    Citation,
    Artifact,
    ToolUse,
    ToolResult,
    Error,
    Complete,
    Cancelled,
    JobStatus,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventType::Message => write!(f, "message"),
            EventType::Reasoning => write!(f, "reasoning"),
            EventType::Citation => write!(f, "citation"),
            EventType::Artifact => write!(f, "artifact"),
            EventType::ToolUse => write!(f, "tool_use"),
            EventType::ToolResult => write!(f, "tool_result"),
            EventType::Error => write!(f, "error"),
            EventType::Complete => write!(f, "complete"),
            EventType::Cancelled => write!(f, "cancelled"),
            EventType::JobStatus => write!(f, "job_status"),
        }
    }
}

// ============================================================================
// Event Data Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageChunkData {
    #[serde(rename = "type")]
    pub data_type: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStepData {
    #[serde(rename = "type")]
    pub data_type: String,
    pub event_type: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationData {
    pub citation_number: i32,
    pub source_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quote: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactData {
    #[serde(rename = "type")]
    pub artifact_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chart_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUseData {
    pub tool_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultData {
    pub tool_id: String,
    pub tool_name: String,
    pub result: serde_json::Value,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorData {
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteData {
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelledData {
    pub job_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatusData {
    pub job_id: String,
    pub file_name: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

// ============================================================================
// SSE Event Wrapper
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SseEvent {
    pub event: String,
    pub data: serde_json::Value,
}

// ============================================================================
// Event Emitter
// ============================================================================

/// Event emitter for outputting SSE-style events to stdout
pub struct EventEmitter {
    mode: OutputMode,
}

impl EventEmitter {
    pub fn new(mode: OutputMode) -> Self {
        Self { mode }
    }

    pub fn json_mode() -> Self {
        Self::new(OutputMode::Json)
    }

    pub fn interactive_mode() -> Self {
        Self::new(OutputMode::Interactive)
    }

    /// Check if emitter is in JSON mode
    pub fn is_json_mode(&self) -> bool {
        matches!(self.mode, OutputMode::Json)
    }

    /// Emit a raw SSE event
    pub fn emit(&self, event_type: EventType, data: serde_json::Value) {
        match self.mode {
            OutputMode::Json => {
                let event = SseEvent {
                    event: event_type.to_string(),
                    data,
                };
                if let Ok(json) = serde_json::to_string(&event) {
                    println!("{}", json);
                    let _ = io::stdout().flush();
                }
            }
            OutputMode::Interactive => {
                self.emit_interactive(event_type, &data);
            }
        }
    }

    /// Emit a message chunk (streaming text content)
    pub fn message_chunk(&self, content: &str) {
        let data = MessageChunkData {
            data_type: "message_chunk".to_string(),
            content: content.to_string(),
        };
        self.emit(
            EventType::Message,
            serde_json::to_value(data).unwrap_or_default(),
        );
    }

    /// Emit a reasoning step (tool invocation info, thinking)
    pub fn reasoning_step(&self, content: &str, event_type: &str) {
        let data = ReasoningStepData {
            data_type: "reasoning_step".to_string(),
            event_type: event_type.to_string(),
            content: content.to_string(),
        };
        self.emit(
            EventType::Reasoning,
            serde_json::to_value(data).unwrap_or_default(),
        );
    }

    /// Emit a citation reference
    pub fn citation(
        &self,
        citation_number: i32,
        source_name: &str,
        page_number: Option<i32>,
        quote: Option<&str>,
    ) {
        let data = CitationData {
            citation_number,
            source_name: source_name.to_string(),
            page_number,
            quote: quote.map(|s| s.to_string()),
        };
        self.emit(
            EventType::Citation,
            serde_json::to_value(data).unwrap_or_default(),
        );
    }

    /// Emit an artifact (chart, table, etc.)
    pub fn artifact(
        &self,
        artifact_type: &str,
        path: Option<&str>,
        content: Option<&str>,
        chart_type: Option<&str>,
    ) {
        let data = ArtifactData {
            artifact_type: artifact_type.to_string(),
            path: path.map(|s| s.to_string()),
            content: content.map(|s| s.to_string()),
            chart_type: chart_type.map(|s| s.to_string()),
        };
        self.emit(
            EventType::Artifact,
            serde_json::to_value(data).unwrap_or_default(),
        );
    }

    /// Emit a tool use event (when LLM invokes a tool)
    pub fn tool_use(&self, tool_id: &str, tool_name: &str, arguments: serde_json::Value) {
        let data = ToolUseData {
            tool_id: tool_id.to_string(),
            tool_name: tool_name.to_string(),
            arguments,
        };
        self.emit(
            EventType::ToolUse,
            serde_json::to_value(data).unwrap_or_default(),
        );
    }

    /// Emit a tool result event
    pub fn tool_result(
        &self,
        tool_id: &str,
        tool_name: &str,
        result: serde_json::Value,
        success: bool,
        error: Option<&str>,
    ) {
        let data = ToolResultData {
            tool_id: tool_id.to_string(),
            tool_name: tool_name.to_string(),
            result,
            success,
            error: error.map(|s| s.to_string()),
        };
        self.emit(
            EventType::ToolResult,
            serde_json::to_value(data).unwrap_or_default(),
        );
    }

    /// Emit an error event
    pub fn error(&self, message: &str, code: Option<&str>) {
        let data = ErrorData {
            message: message.to_string(),
            code: code.map(|s| s.to_string()),
        };
        self.emit(
            EventType::Error,
            serde_json::to_value(data).unwrap_or_default(),
        );
    }

    /// Emit a completion event
    pub fn complete(&self, finish_reason: &str, total_tokens: Option<i32>, model: Option<&str>) {
        let data = CompleteData {
            finish_reason: finish_reason.to_string(),
            total_tokens,
            model: model.map(|s| s.to_string()),
        };
        self.emit(
            EventType::Complete,
            serde_json::to_value(data).unwrap_or_default(),
        );
    }

    /// Emit a cancelled event
    pub fn cancelled(&self, job_id: &str, reason: Option<&str>) {
        let data = CancelledData {
            job_id: job_id.to_string(),
            reason: reason.map(|s| s.to_string()),
        };
        self.emit(
            EventType::Cancelled,
            serde_json::to_value(data).unwrap_or_default(),
        );
    }

    /// Emit a job status event
    pub fn job_status(
        &self,
        job_id: &str,
        file_name: &str,
        status: &str,
        progress: Option<&str>,
        error_message: Option<&str>,
    ) {
        let data = JobStatusData {
            job_id: job_id.to_string(),
            file_name: file_name.to_string(),
            status: status.to_string(),
            progress: progress.map(|s| s.to_string()),
            error_message: error_message.map(|s| s.to_string()),
        };
        self.emit(
            EventType::JobStatus,
            serde_json::to_value(data).unwrap_or_default(),
        );
    }

    /// Emit in interactive mode with colors and formatting
    fn emit_interactive(&self, event_type: EventType, data: &serde_json::Value) {
        match event_type {
            EventType::Message => {
                if let Some(content) = data.get("content").and_then(|v| v.as_str()) {
                    print!("{}", content);
                    let _ = io::stdout().flush();
                }
            }
            EventType::Reasoning => {
                if let Some(content) = data.get("content").and_then(|v| v.as_str()) {
                    let event_subtype = data
                        .get("event_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("INFO");
                    println!("\n\x1b[36m[{}]\x1b[0m {}", event_subtype, content);
                }
            }
            EventType::Citation => {
                if let Some(num) = data.get("citation_number").and_then(|v| v.as_i64()) {
                    let source = data
                        .get("source_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let page = data
                        .get("page_number")
                        .and_then(|v| v.as_i64())
                        .map(|p| format!(", p.{}", p))
                        .unwrap_or_default();
                    println!("\x1b[33m[{}] {}{}\x1b[0m", num, source, page);
                }
            }
            EventType::Artifact => {
                let artifact_type = data
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("artifact");
                if let Some(path) = data.get("path").and_then(|v| v.as_str()) {
                    println!("\n\x1b[35m[{}]\x1b[0m Saved to: {}", artifact_type, path);
                } else {
                    println!("\n\x1b[35m[{}]\x1b[0m Generated", artifact_type);
                }
            }
            EventType::ToolUse => {
                let tool_name = data
                    .get("tool_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let arguments = data.get("arguments");
                println!("\n\x1b[34m🔧 Calling tool:\x1b[0m {}", tool_name);
                if let Some(args) = arguments {
                    // Pretty print the arguments
                    if let Ok(pretty) = serde_json::to_string_pretty(args) {
                        println!(
                            "\x1b[90m   Arguments: {}\x1b[0m",
                            pretty.replace('\n', "\n   ")
                        );
                    }
                }
            }
            EventType::ToolResult => {
                let tool_name = data
                    .get("tool_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let success = data
                    .get("success")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if success {
                    println!("\x1b[32m✓ {} completed\x1b[0m", tool_name);
                } else {
                    let error = data
                        .get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Unknown error");
                    println!("\x1b[31m✗ {} failed: {}\x1b[0m", tool_name, error);
                }
            }
            EventType::Error => {
                let message = data
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error");
                eprintln!("\n\x1b[31mError: {}\x1b[0m", message);
            }
            EventType::Complete => {
                let reason = data
                    .get("finish_reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("stop");
                if reason != "stop" {
                    println!("\n\x1b[90m[Completed: {}]\x1b[0m", reason);
                }
            }
            EventType::Cancelled => {
                let job_id = data
                    .get("job_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                println!("\n\x1b[33mCancelled: {}\x1b[0m", job_id);
            }
            EventType::JobStatus => {
                let file_name = data
                    .get("file_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let status = data
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let progress = data.get("progress").and_then(|v| v.as_str()).unwrap_or("");
                let status_icon = match status {
                    "complete" => "✓",
                    "failed" => "✗",
                    "cancelled" => "⊘",
                    "processing" | "pending" => "⋯",
                    _ => "•",
                };
                let color = match status {
                    "complete" => "\x1b[32m",
                    "failed" => "\x1b[31m",
                    "cancelled" => "\x1b[33m",
                    _ => "\x1b[36m",
                };
                if progress.is_empty() {
                    println!("{}{} {} - {}\x1b[0m", color, status_icon, file_name, status);
                } else {
                    println!(
                        "{}{} {} - {} ({})\x1b[0m",
                        color, status_icon, file_name, status, progress
                    );
                }
            }
        }
    }
}

// ============================================================================
// Global convenience functions
// ============================================================================

/// Create a new event emitter based on JSON flag
pub fn create_emitter(json_mode: bool) -> EventEmitter {
    if json_mode {
        EventEmitter::json_mode()
    } else {
        EventEmitter::interactive_mode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_chunk_json() {
        let emitter = EventEmitter::json_mode();
        // Just verify it doesn't panic
        emitter.message_chunk("Hello, world!");
    }

    #[test]
    fn test_reasoning_step() {
        let emitter = EventEmitter::json_mode();
        emitter.reasoning_step("Executing text2sql", "TOOL");
    }
}
