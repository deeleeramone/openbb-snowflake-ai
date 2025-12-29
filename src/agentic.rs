//! Agentic loop for tool-calling workflows.
//!
//! This module implements the main agentic loop that:
//! 1. Streams LLM responses
//! 2. Detects tool calls in the response
//! 3. Executes tools
//! 4. Continues the conversation with tool results
//! 5. Repeats until no more tool calls or max iterations reached

use crate::agents::{AgentsClient, Message, StreamChunk};
use crate::engine::SnowflakeEngine;
use crate::events::EventEmitter;
use crate::jobs::JobManager;
use crate::tools::ToolExecutor;
use futures::StreamExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::Write;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Maximum number of tool iterations to prevent infinite loops
const MAX_TOOL_ITERATIONS: usize = 15;

/// Maximum total tool calls across all iterations
const MAX_TOTAL_TOOL_CALLS: usize = 25;

/// Build a fresh system prompt with current database context.
/// This is called on every iteration of the agentic loop to ensure
/// the LLM has up-to-date context about the current database, schema, and tables.
async fn build_system_prompt_for_loop(
    engine: &Arc<Mutex<SnowflakeEngine>>,
    tool_executor: &ToolExecutor,
) -> String {
    let guard = engine.lock().await;
    let current_db = guard
        .get_database()
        .await
        .unwrap_or_else(|_| "Not selected".to_string());
    let current_schema = guard
        .get_schema()
        .await
        .unwrap_or_else(|_| "Not selected".to_string());

    // List tables in current schema
    let tables = if !current_db.is_empty() && !current_schema.is_empty() {
        guard
            .list_tables_in(&current_db, &current_schema)
            .await
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    drop(guard);

    // Build table list with fully qualified names
    let table_list = if tables.is_empty() {
        "No tables found or schema not selected.".to_string()
    } else {
        tables
            .iter()
            .map(|t| format!("- {}.{}.{}", current_db, current_schema, t))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let system_prompt = format!(
        r#"You are a Snowflake SQL assistant. Answer user questions by calling tools and presenting results.

## ABSOLUTE RULE: NEVER FABRICATE DATA

🚫 NEVER make up data, examples, or "illustrative" results
🚫 NEVER invent column values, company names, numbers, or any database content
🚫 NEVER show fake results before or instead of querying
🚫 NEVER say "here's an example of what the data might look like"

✅ ONLY show data that comes directly from tool results
✅ If you haven't queried yet, say "Let me query the database" and CALL THE TOOL
✅ If a query returns no results, say "The query returned no results"
✅ EVERY number, name, and value you display MUST come from an actual tool result

If you fabricate any data, you are LYING to the user. This is unacceptable.

## DATABASE CONTEXT
Current: {current_db}.{current_schema}
Tables: {table_list}

## WORKFLOW

1. Get schema with get_table_schema (never guess column names)
2. Execute query with execute_query
3. Present the EXACT results from the tool - no modifications, no additions
4. Answer the question based on the REAL data

## TOOL FORMAT

Output ONLY this JSON:
{{"tool": "tool_name", "arguments": {{"param": "value"}}}}

Examples:
{{"tool": "get_table_schema", "arguments": {{"table_name": "MY_TABLE"}}}}
{{"tool": "execute_query", "arguments": {{"query": "SELECT * FROM {current_db}.{current_schema}.MY_TABLE LIMIT 10"}}}}

## SQL RULES

- Use fully qualified names: {current_db}.{current_schema}.TABLE
- Use double quotes for column names: "Column_Name"
- Check schema before writing queries

Available tools:
{tools}"#,
        current_db = current_db,
        current_schema = current_schema,
        table_list = table_list,
        tools = tool_executor.registry().format_tool_definitions()
    );

    system_prompt
}

/// Parsed tool call from LLM response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Result of running the agentic loop
#[derive(Debug)]
pub struct AgenticResult {
    /// Final response text
    pub response_text: String,
    /// Number of tool calls executed
    pub tool_calls_executed: usize,
    /// Whether the loop was cancelled
    pub cancelled: bool,
    /// Any SQL snippets found in the response
    pub sql_snippets: Vec<String>,
}

/// Parse tool calls from LLM response text.
///
/// Tool calls can appear in several formats:
/// 1. JSON with "tool" field: {"tool": "name", "arguments": {...}}
/// 2. JSON with "function" field: {"function": {"name": "...", "arguments": ...}}
/// 3. Embedded in code blocks
pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut tool_calls = Vec::new();

    // Try to find JSON objects with tool/function fields
    let json_pattern =
        Regex::new(r#"\{[^{}]*"(?:tool|function)"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"#).ok();

    if let Some(pattern) = json_pattern {
        for cap in pattern.find_iter(text) {
            if let Ok(parsed) = serde_json::from_str::<Value>(cap.as_str()) {
                if let Some(tool_call) = extract_tool_call(&parsed) {
                    tool_calls.push(tool_call);
                }
            }
        }
    }

    // Also try parsing as a single JSON object
    if tool_calls.is_empty() {
        if let Some(start) = text.find('{') {
            let remaining = &text[start..];
            // Find matching brace
            let mut brace_count = 0;
            let mut end_idx = 0;
            for (i, c) in remaining.char_indices() {
                match c {
                    '{' => brace_count += 1,
                    '}' => {
                        brace_count -= 1;
                        if brace_count == 0 {
                            end_idx = i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if end_idx > 0 {
                let json_str = &remaining[..end_idx];
                if let Ok(parsed) = serde_json::from_str::<Value>(json_str) {
                    if let Some(tool_call) = extract_tool_call(&parsed) {
                        tool_calls.push(tool_call);
                    }
                }
            }
        }
    }

    // Try parsing XML format: <invoke name="tool_name"><parameter name="key">value</parameter></invoke>
    if tool_calls.is_empty() {
        let xml_pattern = Regex::new(r#"<invoke\s+name="([^"]+)">([\s\S]*?)</invoke>"#).ok();
        if let Some(pattern) = xml_pattern {
            for cap in pattern.captures_iter(text) {
                if let (Some(name_match), Some(params_match)) = (cap.get(1), cap.get(2)) {
                    let tool_name = name_match.as_str().to_string();
                    let params_text = params_match.as_str();

                    // Parse parameters
                    let mut arguments = json!({});
                    let param_pattern =
                        Regex::new(r#"<parameter\s+name="([^"]+)">([^<]*)</parameter>"#).ok();
                    if let Some(pp) = param_pattern {
                        for param_cap in pp.captures_iter(params_text) {
                            if let (Some(key), Some(value)) = (param_cap.get(1), param_cap.get(2)) {
                                let key_str = key.as_str();
                                let value_str = value.as_str().trim();
                                // Try to parse as number or keep as string
                                if let Ok(num) = value_str.parse::<i64>() {
                                    arguments[key_str] = json!(num);
                                } else if let Ok(num) = value_str.parse::<f64>() {
                                    arguments[key_str] = json!(num);
                                } else {
                                    arguments[key_str] = json!(value_str);
                                }
                            }
                        }
                    }

                    tool_calls.push(ToolCall {
                        id: uuid::Uuid::new_v4().to_string(),
                        name: tool_name,
                        arguments,
                    });
                }
            }
        }
    }

    tool_calls
}

/// Extract a ToolCall from a parsed JSON value
fn extract_tool_call(value: &Value) -> Option<ToolCall> {
    // Format 1: {"tool": "name", "arguments": {...}}
    if let Some(tool_name) = value.get("tool").and_then(|v| v.as_str()) {
        let arguments = value.get("arguments").cloned().unwrap_or(json!({}));
        return Some(ToolCall {
            id: uuid::Uuid::new_v4().to_string(),
            name: tool_name.to_string(),
            arguments,
        });
    }

    // Format 2: {"function": {"name": "...", "arguments": ...}}
    if let Some(function) = value.get("function") {
        if let Some(name) = function.get("name").and_then(|v| v.as_str()) {
            let args_str = function.get("arguments");
            let arguments = match args_str {
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(json!({})),
                Some(v) => v.clone(),
                None => json!({}),
            };
            return Some(ToolCall {
                id: value
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                name: name.to_string(),
                arguments,
            });
        }
    }

    None
}

/// Extract SQL snippets from text
pub fn extract_sql_snippets(text: &str) -> Vec<String> {
    let mut snippets = Vec::new();
    let sql_block_re = Regex::new(r"```sql\s*\n([\s\S]*?)```").ok();

    if let Some(re) = sql_block_re {
        for cap in re.captures_iter(text) {
            if let Some(sql) = cap.get(1) {
                let sql_text = sql.as_str().trim().to_string();
                if !sql_text.is_empty() {
                    snippets.push(sql_text);
                }
            }
        }
    }

    snippets
}

/// Run the agentic loop for a user query.
///
/// This function:
/// 1. Sends the user message to the LLM
/// 2. Streams the response live to user, detecting tool calls
/// 3. When a tool call is detected, executes it immediately
/// 4. Adds tool result to messages and continues
/// 5. Repeats until LLM gives a final answer without tool calls
pub async fn run_agentic_loop(
    engine: Arc<Mutex<SnowflakeEngine>>,
    agent_client: &mut AgentsClient,
    tool_executor: &ToolExecutor,
    emitter: &EventEmitter,
    _job_manager: &mut JobManager,
    model: &str,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
    initial_messages: Vec<Message>,
    cancel_token: Option<CancellationToken>,
) -> Result<AgenticResult, String> {
    let mut messages = initial_messages;
    let mut total_tool_calls = 0;
    let mut iteration = 0;
    let mut final_response = String::new();

    // Main agentic loop
    loop {
        iteration += 1;

        if iteration > MAX_TOOL_ITERATIONS {
            if !emitter.is_json_mode() {
                println!(
                    "\n⚠ Maximum iterations ({}) reached. Stopping.",
                    MAX_TOOL_ITERATIONS
                );
            }
            emitter.error("Maximum tool iterations reached.", Some("MAX_ITERATIONS"));
            break;
        }

        if total_tool_calls >= MAX_TOTAL_TOOL_CALLS {
            if !emitter.is_json_mode() {
                println!(
                    "\n⚠ Maximum tool calls ({}) reached. Stopping.",
                    MAX_TOTAL_TOOL_CALLS
                );
            }
            emitter.error("Maximum total tool calls reached.", Some("MAX_TOOL_CALLS"));
            break;
        }

        // Check for cancellation
        if let Some(ref token) = cancel_token {
            if token.is_cancelled() {
                emitter.cancelled("user", Some("User cancelled operation"));
                return Ok(AgenticResult {
                    response_text: final_response,
                    tool_calls_executed: total_tool_calls,
                    cancelled: true,
                    sql_snippets: Vec::new(),
                });
            }
        }

        // First iteration: show analyzing message
        if iteration == 1 {
            emitter.reasoning_step(
                "Analyzing request and determining required tools...",
                "INFO",
            );
        }

        // Build fresh system prompt with current database context
        let system_prompt = build_system_prompt_for_loop(&engine, tool_executor).await;

        // Prepare messages with fresh system prompt
        let mut request_messages = vec![Message::new_system(system_prompt)];
        for msg in &messages {
            if msg.role.as_deref() != Some("system") {
                request_messages.push(msg.clone());
            }
        }

        // Stream LLM response
        let stream_result = agent_client
            .stream_complete(model, request_messages, temperature, None, max_tokens, None)
            .await;

        let (mut stream, metadata) = match stream_result {
            Ok(result) => result,
            Err(e) => {
                emitter.error(
                    &format!("Failed to get LLM response: {}", e),
                    Some("LLM_ERROR"),
                );
                return Err(e);
            }
        };

        // Accumulate the full response for this iteration
        let mut iteration_text = String::new();
        let mut streamed_text = String::new();
        let mut started_streaming = false;
        let mut in_json_block = false;
        let mut json_block_depth = 0;

        // Stream chunks LIVE to the user, but filter out tool call JSON
        while let Some(chunk_result) = stream.next().await {
            if let Some(ref token) = cancel_token {
                if token.is_cancelled() {
                    emitter.cancelled("user", Some("User cancelled operation"));
                    return Ok(AgenticResult {
                        response_text: final_response,
                        tool_calls_executed: total_tool_calls,
                        cancelled: true,
                        sql_snippets: Vec::new(),
                    });
                }
            }

            match chunk_result {
                Ok(chunk) => {
                    for choice in &chunk.choices {
                        if let Some(content) = &choice.delta.text {
                            // Always accumulate full text for tool call detection
                            iteration_text.push_str(content);

                            // For streaming, filter out JSON tool blocks
                            // Track ```json blocks and standalone { "tool": } objects
                            let mut filtered = String::new();
                            for ch in content.chars() {
                                // Detect start of JSON block
                                if streamed_text.ends_with("```json")
                                    || streamed_text.ends_with("```JSON")
                                {
                                    in_json_block = true;
                                    // Remove the ```json we already added
                                    for _ in 0..7 {
                                        streamed_text.pop();
                                        filtered.pop();
                                    }
                                }

                                // Detect end of JSON block
                                if in_json_block
                                    && streamed_text.ends_with("```")
                                    && !streamed_text.ends_with("````")
                                {
                                    in_json_block = false;
                                    // Remove the closing ```
                                    for _ in 0..3 {
                                        streamed_text.pop();
                                    }
                                    continue;
                                }

                                // Track braces for standalone JSON
                                if ch == '{' && !in_json_block {
                                    // Check if this looks like a tool call
                                    let recent: String = streamed_text
                                        .chars()
                                        .rev()
                                        .take(50)
                                        .collect::<String>()
                                        .chars()
                                        .rev()
                                        .collect();
                                    if recent.contains("\"tool\"")
                                        || streamed_text.ends_with("{\"tool\"")
                                    {
                                        json_block_depth += 1;
                                    }
                                }
                                if ch == '}' && json_block_depth > 0 {
                                    json_block_depth -= 1;
                                }

                                streamed_text.push(ch);

                                // Only add to filtered output if not in a JSON block
                                if !in_json_block && json_block_depth == 0 {
                                    filtered.push(ch);
                                }
                            }

                            // Stream the filtered content
                            if !filtered.is_empty() && !emitter.is_json_mode() {
                                if !started_streaming {
                                    print!("\n");
                                    started_streaming = true;
                                }
                                print!("{}", filtered);
                                std::io::stdout().flush().ok();
                            } else if !filtered.is_empty() {
                                emitter.message_chunk(&filtered);
                            }
                        }
                    }
                }
                Err(e) => {
                    emitter.error(&format!("Stream error: {}", e), Some("STREAM_ERROR"));
                    break;
                }
            }
        }

        // Add newline after streaming completes
        if started_streaming && !emitter.is_json_mode() {
            println!();
        }

        // Update metadata
        agent_client.update_metadata_and_stats(metadata);

        // Parse tool calls from the response
        let tool_calls = parse_tool_calls(&iteration_text);

        if tool_calls.is_empty() {
            // No tool calls - this is the final answer!
            // Text was already streamed live, just save it
            final_response = iteration_text.clone();

            // Done - we have our answer
            break;
        }

        // There are tool calls - execute them
        // First, add the assistant message (without tool JSON) to history
        let clean_text = strip_tool_json(&iteration_text);
        if !clean_text.trim().is_empty() {
            messages.push(Message::new_assistant(clean_text.clone()));
        } else {
            // Even if empty, we need an assistant message for proper alternation
            messages.push(Message::new_assistant(
                "I'll use the following tools to help answer your question.".to_string(),
            ));
        }

        // Execute each tool call and collect results
        let mut tool_results: Vec<String> = Vec::new();

        for tool_call in &tool_calls {
            total_tool_calls += 1;

            // Emit tool_use event (shows the "Calling tool" message)
            emitter.tool_use(&tool_call.id, &tool_call.name, tool_call.arguments.clone());

            // Execute the tool
            let result = tool_executor
                .execute(
                    &engine,
                    &tool_call.name,
                    tool_call.arguments.clone(),
                    emitter,
                )
                .await;

            // Format result for display and for LLM
            let result_text = if result.success {
                let result_str = serde_json::to_string_pretty(&result.result)
                    .unwrap_or_else(|_| result.result.to_string());
                format!("[Tool Result: {}]\n{}", tool_call.name, result_str)
            } else {
                let err_msg = result.error.as_deref().unwrap_or("Unknown error");
                format!("[Tool Error: {}]\n{}", tool_call.name, err_msg)
            };

            // Emit tool_result event (shows success/failure)
            emitter.tool_result(
                &tool_call.id,
                &tool_call.name,
                result.result.clone(),
                result.success,
                result.error.as_deref(),
            );

            tool_results.push(result_text);
        }

        // Add all tool results as a single user message
        if !tool_results.is_empty() {
            let combined_results = tool_results.join("\n\n");
            messages.push(Message::new_user(combined_results));
        }

        // Continue loop - LLM will see tool results and respond
    }

    // Extract any SQL snippets from the final response
    let sql_snippets = extract_sql_snippets(&final_response);

    emitter.complete("stop", None, Some(model));

    Ok(AgenticResult {
        response_text: final_response,
        tool_calls_executed: total_tool_calls,
        cancelled: false,
        sql_snippets,
    })
}

/// Strip tool call JSON blocks from text for cleaner conversation history
fn strip_tool_json(text: &str) -> String {
    let mut result = text.to_string();

    // Remove ```json {...} ``` blocks
    let json_block_re = Regex::new(r"```json\s*\{[^`]*\}\s*```").ok();
    if let Some(re) = json_block_re {
        result = re.replace_all(&result, "").to_string();
    }

    // Remove standalone {"tool": ...} JSON objects
    let standalone_re =
        Regex::new(r#"\{"tool"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}"#).ok();
    if let Some(re) = standalone_re {
        result = re.replace_all(&result, "").to_string();
    }

    // Clean up extra whitespace
    result = result.trim().to_string();

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_calls_json_format() {
        let text =
            r#"I'll run the query. {"tool": "execute_query", "arguments": {"sql": "SELECT 1"}}"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "execute_query");
    }

    #[test]
    fn test_parse_tool_calls_function_format() {
        let text = r#"{"id": "call_1", "function": {"name": "text2sql", "arguments": "{\"query\": \"show tables\"}"}}"#;
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "text2sql");
    }

    #[test]
    fn test_extract_sql_snippets() {
        let text = "Here's the SQL:\n```sql\nSELECT * FROM users;\n```\nDone!";
        let snippets = extract_sql_snippets(text);
        assert_eq!(snippets.len(), 1);
        assert_eq!(snippets[0], "SELECT * FROM users;");
    }
}
