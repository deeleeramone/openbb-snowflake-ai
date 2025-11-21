use serde::{Deserialize, Serialize};
use std::pin::Pin;
use futures::Stream;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Request Types (matching Snowflake API exactly)
// ============================================================================

use pyo3::prelude::*;
// Removed `Python` from import as it's unused.

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_list: Option<Vec<ContentItem>>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContentItem {
    #[pyo3(get, set)]
    #[serde(rename = "type")]
    pub content_type: String,
    #[pyo3(get, set)]
    pub text: String,
}

#[pymethods]
impl Message {
    #[pyo3(name = "get_content")]
    pub fn get_content(&self) -> String {
        self.content.clone()
            .or_else(|| {
                self.content_list.as_ref().and_then(|list| {
                    list.first().map(|item| item.text.clone())
                })
            })
            .unwrap_or_default()
    }
}

// These are internal Rust helper functions, not Python methods.
impl Message {
    pub fn new_user(content: String) -> Self {
        Self {
            role: Some("user".to_string()),
            content: Some(content),
            content_list: None,
        }
    }

    pub fn new_assistant(content: String) -> Self {
        Self {
            role: Some("assistant".to_string()),
            content: Some(content),
            content_list: None,
        }
    }

    pub fn new_system(content: String) -> Self {
        Self {
            role: Some("system".to_string()),
            content: Some(content),
            content_list: None,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CompleteRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    pub stream: bool,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Tool {
    #[pyo3(get, set)]
    pub tool_spec: ToolSpec,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolSpec {
    #[pyo3(get, set)]
    #[serde(rename = "type")]
    pub tool_type: String,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get, set)]
    pub input_schema: ToolInputSchema,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolInputSchema {
    #[pyo3(get, set)]
    #[serde(rename = "type")]
    pub schema_type: String,
    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, ToolInputSchema>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<ToolInputSchema>>,
    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}







#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolResource {
    // text2sql specific fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_model_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_view: Option<String>,

    // search specific fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_service: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title_column: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id_column: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<serde_json::Value>,

    // generic (function) specific fields
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub resource_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identifier: Option<String>,

    // Common field for text2sql and generic functions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_environment: Option<ExecutionEnvironment>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExecutionEnvironment {
    #[serde(rename = "type")]
    pub env_type: String, // "warehouse" or other types
    pub warehouse: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_timeout: Option<i32>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum ToolChoice {
    Auto(String),
    Required(String),
    None(String),
    Specific {
        #[serde(rename = "type")]
        tool_type: String,
        function: FunctionName
    },
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionName {
    pub name: String,
}

// ============================================================================
// Response Types (matching Snowflake API exactly)
// ============================================================================

#[pyclass]
#[derive(Debug, Deserialize, Clone)]
pub struct CompleteResponse {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub object: String,
    #[pyo3(get, set)]
    pub created: i64,
    #[pyo3(get, set)]
    pub model: String,
    #[pyo3(get, set)]
    pub choices: Vec<Choice>,
    #[pyo3(get, set)]
    pub usage: Usage,
}

#[pyclass]
#[derive(Debug, Deserialize, Clone)]
pub struct Choice {
    #[pyo3(get, set)]
    pub index: i32,
    #[pyo3(get, set)]
    pub message: Message,
    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    #[serde(rename = "type")]
    pub tool_type: String,
    #[pyo3(get, set)]
    pub function: FunctionCall,
}

#[pymethods]
impl ToolCall {
    #[new]
    fn new(id: String, tool_type: String, function: FunctionCall) -> Self {
        Self { id, tool_type, function }
    }
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub arguments: String,
}

#[pymethods]
impl FunctionCall {
    #[new]
    fn new(name: String, arguments: String) -> Self {
        Self { name, arguments }
    }
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Usage {
    #[pyo3(get, set)]
    #[serde(default)]
    pub prompt_tokens: i32,
    #[pyo3(get, set)]
    #[serde(default)]
    pub completion_tokens: i32,
    #[pyo3(get, set)]
    #[serde(default)]
    pub total_tokens: i32,
}

// ============================================================================
// Streaming Response Types (matching Snowflake API exactly)
// ============================================================================

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamChunk {
    #[pyo3(get, set)]
    pub id: String,
    #[pyo3(get, set)]
    pub model: String,
    #[pyo3(get, set)]
    pub choices: Vec<StreamChoice>,
    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[pymethods]
impl StreamChunk {
    fn to_json_string(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| PyErr::new::<PyAny, _>(format!("Failed to serialize StreamChunk: {}", e)))
    }
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamChoice {
    #[pyo3(get, set)]
    pub delta: Delta,
    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Delta {
    #[pyo3(get, set)]
    #[serde(rename = "type")]
    pub delta_type: String,

    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<String>,

    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,

    #[pyo3(get, set)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

// ============================================================================
// Metadata and Statistics Types
// ============================================================================

#[pyclass]
#[derive(Debug, Clone)]
pub struct CompletionMetadata {
    pub request_id: String,
    pub model: String,
    pub created: i64,
    pub object: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Default)]
pub struct UsageStats {
    pub total_requests: usize,
    pub total_prompt_tokens: i64,
    pub total_completion_tokens: i64,
    pub total_tokens: i64,
    pub requests_by_model: HashMap<String, usize>,
}

impl UsageStats {
    pub fn add_usage(&mut self, model: &str, usage: &Usage) {
        self.total_requests += 1;
        self.total_prompt_tokens += usage.prompt_tokens as i64;
        self.total_completion_tokens += usage.completion_tokens as i64;
        self.total_tokens += usage.total_tokens as i64;
        *self.requests_by_model.entry(model.to_string()).or_insert(0) += 1;
    }
}

// ============================================================================
// Agent CRUD Types (Snowflake Cortex Agents REST API) - CORRECT TYPES
// ============================================================================

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentProfile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avatar_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentInstructions {
    pub response: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orchestration: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_questions: Option<Vec<String>>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BudgetConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seconds: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<i32>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OrchestrationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget: Option<BudgetConfig>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Agent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<AgentProfile>,
    pub models: ModelConfig,
    pub instructions: AgentInstructions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orchestration: Option<OrchestrationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_resources: Option<HashMap<String, ToolResource>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_on: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_on: Option<i64>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CreateAgentRequest {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<AgentProfile>,
    pub models: ModelConfig,
    pub instructions: AgentInstructions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orchestration: Option<OrchestrationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_resources: Option<HashMap<String, ToolResource>>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UpdateAgentRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<AgentProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<ModelConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<AgentInstructions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orchestration: Option<OrchestrationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_resources: Option<HashMap<String, ToolResource>>,
}

// ============================================================================
// Agent Session Types (Snowflake Cortex Agents REST API)
// ============================================================================

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentSession {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub agent_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_on: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub updated_on: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CreateSessionRequest {
    pub agent_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[pyclass]
#[derive(Debug, Deserialize, Clone)]
pub struct ListSessionsResponse {
    pub sessions: Vec<AgentSession>,
}

// ============================================================================
// Agent Message Management
// ============================================================================

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub session_id: String,
    pub role: String,
    pub content: Vec<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_on: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum MessageContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageUrl {
    pub url: String,
}

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SendMessageRequest {
    pub session_id: String,
    pub content: Vec<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[pyclass]
#[derive(Debug, Deserialize, Clone)]
pub struct MessageResponse {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: Vec<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_on: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[pyclass]
#[derive(Debug, Deserialize, Clone)]
pub struct ListMessagesResponse {
    pub messages: Vec<AgentMessage>,
}

// ============================================================================
// Cache-related Types
// ============================================================================

#[pyclass]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CachedMessage {
    #[pyo3(get, set)]
    pub role: String,
    #[pyo3(get, set)]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

#[pymethods]
impl CachedMessage {
    #[new]
    fn new(role: String, content: String, details_str: Option<String>) -> PyResult<Self> {
        let details = match details_str {
            Some(s) => Some(
                serde_json::from_str(&s)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
            ),
            None => None,
        };
        Ok(Self {
            role,
            content,
            details,
        })
    }

    #[getter]
    fn get_details(&self) -> PyResult<Option<String>> {
        match &self.details {
            Some(v) => Ok(Some(serde_json::to_string(v).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?)),
            None => Ok(None),
        }
    }

    #[setter]
    fn set_details(&mut self, details_str: Option<String>) -> PyResult<()> {
        self.details = match details_str {
            Some(s) => Some(
                serde_json::from_str(&s)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?,
            ),
            None => None,
        };
        Ok(())
    }
}

// ============================================================================
// Client Implementation
// ============================================================================

#[pyclass]
#[derive(Clone)]
pub struct AgentsClient {
    account: String,
    token: String,
    database: String,
    schema: String,
    conversation_history: Vec<Message>,
    usage_stats: UsageStats,
    last_metadata: Option<CompletionMetadata>,
}

impl AgentsClient {
    pub fn new(account: String, token: String, database: String, schema: String) -> Self {
        Self {
            account,
            token,
            database,
            schema,
            conversation_history: Vec::new(),
            usage_stats: UsageStats::default(),
            last_metadata: None,
        }
    }

    pub fn add_assistant_response(&mut self, content: String) {
        self.conversation_history.push(Message::new_assistant(content));
    }

    pub fn update_metadata_and_stats(&mut self, metadata: CompletionMetadata) {
        if let Some(usage) = &metadata.usage {
            self.usage_stats.add_usage(&metadata.model, usage);
        }
        self.last_metadata = Some(metadata);
    }

    pub fn reset_conversation(&mut self) {
        self.conversation_history.clear();
    }

    pub fn get_conversation_history(&self) -> Vec<Message> {
        self.conversation_history.clone()
    }

    pub fn add_system_message(&mut self, content: &str) {
        self.conversation_history.insert(0, Message {
            role: Some("system".to_string()),
            content: Some(content.to_string()),
            content_list: None,
        });
    }

    pub fn get_usage_stats(&self) -> UsageStats {
        self.usage_stats.clone()
    }

    pub fn reset_usage_stats(&mut self) {
        self.usage_stats = UsageStats::default();
    }

    pub fn get_last_metadata(&self) -> Option<CompletionMetadata> {
        self.last_metadata.clone()
    }

    pub fn get_usage_report(&self) -> String {
        let stats = &self.usage_stats;
        let mut report = format!(
            "Total Requests: {}\n\
             Total Prompt Tokens: {}\n\
             Total Completion Tokens: {}\n\
             Total Tokens: {}\n\n\
             Requests by Model:\n",
            stats.total_requests,
            stats.total_prompt_tokens,
            stats.total_completion_tokens,
            stats.total_tokens
        );

        for (model, count) in &stats.requests_by_model {
            report.push_str(&format!("  {}: {} requests\n", model, count));
        }

        if let Some(metadata) = &self.last_metadata {
            report.push_str(&format!("\nLast Request:\n"));
            report.push_str(&format!("  ID: {}\n", metadata.request_id));
            report.push_str(&format!("  Model: {}\n", metadata.model));
            report.push_str(&format!("  Object: {}\n", metadata.object));
            report.push_str(&format!("  Created: {}\n", metadata.created));
            if let Some(usage) = &metadata.usage {
                report.push_str(&format!("  Usage: {} prompt + {} completion = {} total tokens\n",
                    usage.prompt_tokens, usage.completion_tokens, usage.total_tokens));
            }
        }

        report
    }

    pub async fn stream_complete(
        &mut self,
        model: &str,
        messages: Vec<Message>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<i32>,
        tools: Option<Vec<Tool>>,
    ) -> Result<(Pin<Box<dyn Stream<Item = Result<StreamChunk, String>> + Send>>, CompletionMetadata), String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/cortex/inference:complete",
            self.account
        );

        let request_body = CompleteRequest {
            model: model.to_string(),
            messages,
            temperature,
            top_p,
            max_tokens,
            stop: None,
            tools,
            tool_choice: None,
            stream: true,
        };

        let request_body_json = serde_json::to_string_pretty(&request_body).unwrap_or_else(|e| format!("Failed to serialize request body: {}", e));
        if std::env::var("SNOWFLAKE_DEBUG").is_ok() {
            println!("[CORTEX REQUEST BODY]:\n{}", request_body_json);
        }

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .header("X-Snowflake-Database", &self.database)
            .header("X-Snowflake-Schema", &self.schema)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Cortex API request failed: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Cortex API error {}: {}", status, error_text));
        }

        let metadata_ref = Arc::new(tokio::sync::Mutex::new(CompletionMetadata {
            request_id: String::new(),
            model: model.to_string(),
            created: chrono::Utc::now().timestamp(),
            object: "chat.completion.chunk".to_string(),
            usage: None,
        }));

        let metadata_clone = metadata_ref.clone();
        let byte_stream = response.bytes_stream();
        let event_stream = byte_stream.eventsource();

        let processed_stream = event_stream.filter_map(move |event_result| {
            let metadata_ref = metadata_ref.clone();
            async move {
                match event_result {
                    Ok(event) => {
                        if event.data == "[DONE]" {
                            return None;
                        }
                        if std::env::var("SNOWFLAKE_DEBUG").is_ok() {
                            println!("[RAW SSE DATA]: {}", &event.data);
                        }

                        match serde_json::from_str::<StreamChunk>(&event.data) {
                            Ok(chunk) => {
                                let mut meta = metadata_ref.lock().await;
                                if meta.request_id.is_empty() {
                                    meta.request_id = chunk.id.clone();
                                    meta.model = chunk.model.clone();
                                }
                                if chunk.usage.is_some() {
                                    meta.usage = chunk.usage.clone();
                                }
                                drop(meta);

                                Some(Ok(chunk))
                            }
                            Err(e) => Some(Err(format!("Failed to parse SSE chunk: {}", e))),
                        }
                    }
                    Err(e) => Some(Err(format!("Stream error: {}", e))),
                }
            }
        });

        let final_metadata = metadata_clone.lock().await.clone();

        if let Some(usage) = &final_metadata.usage {
            self.usage_stats.add_usage(&final_metadata.model, usage);
        }
        self.last_metadata = Some(final_metadata.clone());

        Ok((Box::pin(processed_stream), final_metadata))
    }

    pub async fn complete(
        &mut self,
        model: &str,
        message: &str,
        use_history: bool,
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<i32>,
    ) -> Result<CompleteResponse, String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/cortex/inference:complete",
            self.account
        );

        let mut messages = if use_history {
            self.conversation_history.clone()
        } else {
            Vec::new()
        };

        messages.push(Message {
            role: Some("user".to_string()),
            content: Some(message.to_string()),
            content_list: None,
        });

        let request_body = CompleteRequest {
            model: model.to_string(),
            messages: messages.clone(),
            temperature,
            top_p,
            max_tokens,
            stop: None,
            tools: None,
            tool_choice: None,
            stream: false,
        };

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .header("X-Snowflake-Database", &self.database)
            .header("X-Snowflake-Schema", &self.schema)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Cortex API request failed: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Cortex API error {}: {}", status, error_text));
        }

        let response_text = response
            .text()
            .await
            .map_err(|e| format!("Failed to read response: {}", e))?;

        let complete_response: CompleteResponse = serde_json::from_str(&response_text)
            .map_err(|e| format!("Failed to parse response: {}. Body: {}", e, response_text))?;

        self.usage_stats.add_usage(&complete_response.model, &complete_response.usage);
        self.last_metadata = Some(CompletionMetadata {
            request_id: complete_response.id.clone(),
            model: complete_response.model.clone(),
            created: complete_response.created,
            object: complete_response.object.clone(),
            usage: Some(complete_response.usage.clone()),
        });

        if use_history {
            self.conversation_history.push(Message::new_user(message.to_string()));

            if let Some(choice) = complete_response.choices.first() {
                self.conversation_history.push(Message::new_assistant(choice.message.get_content()));
            }
        }

        Ok(complete_response)
    }

    pub async fn complete_with_tools(
        &mut self,
        model: &str,
        message: &str,
        use_history: bool,
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<i32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<CompleteResponse, String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/cortex/inference:complete",
            self.account
        );

        let mut messages = if use_history {
            self.conversation_history.clone()
        } else {
            Vec::new()
        };

        messages.push(Message {
            role: Some("user".to_string()),
            content: Some(message.to_string()),
            content_list: None,
        });

        let request_body = CompleteRequest {
            model: model.to_string(),
            messages: messages.clone(),
            temperature,
            top_p,
            max_tokens,
            stop: None,
            tools,
            tool_choice,
            stream: false,
        };

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .header("X-Snowflake-Database", &self.database)
            .header("X-Snowflake-Schema", &self.schema)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Cortex API request failed: {}", e))?;
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Cortex API error {}: {}", status, error_text));
        }
        let response_text = response
            .text()
            .await
            .map_err(|e| format!("Failed to read response: {}", e))?;
        let complete_response: CompleteResponse = serde_json::from_str(&response_text)
            .map_err(|e| format!("Failed to parse response: {}. Body: {}", e, response_text))?;

        self.usage_stats.add_usage(&complete_response.model, &complete_response.usage);
        self.last_metadata = Some(CompletionMetadata {
            request_id: complete_response.id.clone(),
            model: complete_response.model.clone(),
            created: complete_response.created,
            object: complete_response.object.clone(),
            usage: Some(complete_response.usage.clone()),
        });

        if use_history {
            self.conversation_history.push(Message::new_user(message.to_string()));
            if let Some(choice) = complete_response.choices.first() {
                self.conversation_history.push(Message::new_assistant(choice.message.get_content()));
            }
        }

        Ok(complete_response)
    }

    pub async fn create_agent(
        &self,
        request: CreateAgentRequest,
        create_mode: Option<&str>,
    ) -> Result<Agent, String> {
        let mut url = format!(
            "https://{}.snowflakecomputing.com/api/v2/databases/{}/schemas/{}/agents",
            self.account, self.database, self.schema
        );

        if let Some(mode) = create_mode {
            url.push_str(&format!("?createMode={}", mode));
        }

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Failed to create agent: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Agent creation failed {}: {}", status, error_text));
        }

        let agent: Agent = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse agent response: {}", e))?;
        Ok(agent)
    }

    pub async fn get_agent(&self, agent_name: &str) -> Result<Agent, String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/databases/{}/schemas/{}/agents/{}",
            self.account, self.database, self.schema, agent_name
        );

        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .send()
            .await
            .map_err(|e| format!("Failed to get agent: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Failed to get agent {}: {}", status, error_text));
        }

        let agent: Agent = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse agent: {}", e))?;
        Ok(agent)
    }

    pub async fn list_agents(
        &self,
        like: Option<&str>,
        from_name: Option<&str>,
        show_limit: Option<i32>,
    ) -> Result<Vec<Agent>, String> {
        let mut url = format!(
            "https://{}.snowflakecomputing.com/api/v2/databases/{}/schemas/{}/agents",
            self.account, self.database, self.schema
        );

        let mut query_params = Vec::new();
        if let Some(pattern) = like {
            query_params.push(format!("like={}", urlencoding::encode(pattern)));
        }
        if let Some(name) = from_name {
            query_params.push(format!("fromName={}", urlencoding::encode(name)));
        }
        if let Some(limit) = show_limit {
            let clamped_limit = limit.max(1).min(10000);
            query_params.push(format!("showLimit={}", clamped_limit));
        }
        if !query_params.is_empty() {
            url.push('?');
            url.push_str(&query_params.join("&"));
        }

        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .send()
            .await
            .map_err(|e| format!("Failed to list agents: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Failed to list agents {}: {}", status, error_text));
        }

        let agents: Vec<Agent> = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse agents list: {}", e))?;
        Ok(agents)
    }

    pub async fn update_agent(
        &self,
        agent_name: &str,
        request: UpdateAgentRequest,
    ) -> Result<Agent, String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/databases/{}/schemas/{}/agents/{}",
            self.account, self.database, self.schema, agent_name
        );

        let client = reqwest::Client::new();
        let response = client
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Failed to update agent: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Failed to update agent {}: {}", status, error_text));
        }

        let agent: Agent = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse updated agent: {}", e))?;
        Ok(agent)
    }

    pub async fn delete_agent(&self, agent_name: &str, if_exists: Option<bool>) -> Result<(), String> {
        let mut url = format!(
            "https://{}.snowflakecomputing.com/api/v2/databases/{}/schemas/{}/agents/{}",
            self.account, self.database, self.schema, agent_name
        );

        if let Some(if_exists_value) = if_exists {
            url.push_str(&format!("?ifExists={}", if_exists_value));
        }

        let client = reqwest::Client::new();
        let response = client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .send()
            .await
            .map_err(|e| format!("Failed to delete agent: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Failed to delete agent {}: {}", status, error_text));
        }

        Ok(())
    }

    pub async fn create_session(
        &self,
        agent_id: &str,
        metadata: Option<serde_json::Value>,
    ) -> Result<AgentSession, String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/cortex/agents/sessions",
            self.account
        );

        let request = CreateSessionRequest {
            agent_id: agent_id.to_string(),
            metadata,
        };

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .header("X-Snowflake-Database", &self.database)
            .header("X-Snowflake-Schema", &self.schema)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Failed to create session: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Session creation failed {}: {}", status, error_text));
        }

        let session: AgentSession = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse session response: {}", e))?;
        Ok(session)
    }

    pub async fn get_session(&self, session_id: &str) -> Result<AgentSession, String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/cortex/agents/sessions/{}",
            self.account, session_id
        );

        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .header("X-Snowflake-Database", &self.database)
            .header("X-Snowflake-Schema", &self.schema)
            .send()
            .await
            .map_err(|e| format!("Failed to get session: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Failed to get session {}: {}", status, error_text));
        }

        let session: AgentSession = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse session: {}", e))?;
        Ok(session)
    }

    pub async fn list_sessions(&self, agent_id: &str) -> Result<Vec<AgentSession>, String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/cortex/agents/sessions?agent_id={}",
            self.account, agent_id
        );

        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .header("X-Snowflake-Database", &self.database)
            .header("X-Snowflake-Schema", &self.schema)
            .send()
            .await
            .map_err(|e| format!("Failed to list sessions: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Failed to list sessions {}: {}", status, error_text));
        }

        let list_response: ListSessionsResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse sessions list: {}", e))?;
        Ok(list_response.sessions)
    }

    pub async fn delete_session(&self, session_id: &str) -> Result<(), String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/cortex/agents/sessions/{}",
            self.account, session_id
        );

        let client = reqwest::Client::new();
        let response = client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .header("X-Snowflake-Database", &self.database)
            .header("X-Snowflake-Schema", &self.schema)
            .send()
            .await
            .map_err(|e| format!("Failed to delete session: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Failed to delete session {}: {}", status, error_text));
        }

        Ok(())
    }

    pub async fn send_message(&self, session_id: &str, content: &str) -> Result<MessageResponse, String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/cortex/agents/sessions/{}/messages",
            self.account, session_id
        );

        let request = SendMessageRequest {
            session_id: session_id.to_string(),
            content: vec![MessageContent::Text {
                text: content.to_string(),
            }],
            stream: Some(false),
        };

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .header("X-Snowflake-Database", &self.database)
            .header("X-Snowflake-Schema", &self.schema)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Failed to send message: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Failed to send message {}: {}", status, error_text));
        }

        let message: MessageResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse message response: {}", e))?;
        Ok(message)
    }

    pub async fn stream_message(
        &self,
        session_id: &str,
        content: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, String>> + Send>>, String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/cortex/agents/sessions/{}/messages",
            self.account, session_id
        );

        let request = SendMessageRequest {
            session_id: session_id.to_string(),
            content: vec![MessageContent::Text {
                text: content.to_string(),
            }],
            stream: Some(true),
        };

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .header("X-Snowflake-Database", &self.database)
            .header("X-Snowflake-Schema", &self.schema)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("Failed to send message: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Failed to send message {}: {}", status, error_text));
        }

        let byte_stream = response.bytes_stream();
        let event_stream = byte_stream.eventsource();

        let processed_stream = event_stream.filter_map(|event_result| async move {
            match event_result {
                Ok(event) => {
                    if event.data == "[DONE]" {
                        return None;
                    }

                    match serde_json::from_str::<StreamChunk>(&event.data) {
                        Ok(chunk) => Some(Ok(chunk)),
                        Err(e) => Some(Err(format!("Failed to parse SSE chunk: {}", e))),
                    }
                }
                Err(e) => Some(Err(format!("Stream error: {}", e))),
            }
        });

        Ok(Box::pin(processed_stream))
    }

    pub async fn list_messages(&self, session_id: &str) -> Result<Vec<AgentMessage>, String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/cortex/agents/sessions/{}/messages",
            self.account, session_id
        );

        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .header("X-Snowflake-Database", &self.database)
            .header("X-Snowflake-Schema", &self.schema)
            .send()
            .await
            .map_err(|e| format!("Failed to list messages: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Failed to list messages {}: {}", status, error_text));
        }

        let list_response: ListMessagesResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse messages list: {}", e))?;
        Ok(list_response.messages)
    }

    pub async fn get_message(&self, session_id: &str, message_id: &str) -> Result<AgentMessage, String> {
        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/cortex/agents/sessions/{}/messages/{}",
            self.account, session_id, message_id
        );

        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .header("X-Snowflake-Database", &self.database)
            .header("X-Snowflake-Schema", &self.schema)
            .send()
            .await
            .map_err(|e| format!("Failed to get message: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("Failed to get message {}: {}", status, error_text));
        }

        let message: AgentMessage = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse message: {}", e))?;
        Ok(message)
    }

    pub async fn create_agent_from_file(
        &self,
        file_path: &str,
        create_mode: Option<&str>,
    ) -> Result<Agent, String> {
        let json_content = std::fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let create_req: CreateAgentRequest = serde_json::from_str(&json_content)
            .map_err(|e| format!("Invalid JSON format: {}", e))?;

        self.create_agent(create_req, create_mode).await
    }
}
