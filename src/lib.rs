#![cfg(feature = "extension-module")]

pub mod agents;
pub mod engine;

use crate::agents::Message;
use futures::StreamExt;
use pyo3::prelude::*;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::agents::StreamChunk;
use crate::engine::SnowflakeEngine;

#[pyclass]
struct SnowflakeAI {
    engine: Arc<tokio::sync::Mutex<SnowflakeEngine>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl SnowflakeAI {
    #[new]
    fn new(
        user: String,
        password: String,
        account: String,
        role: String,
        warehouse: String,
        database: String,
        schema: String,
    ) -> PyResult<Self> {
        let runtime = Arc::new(tokio::runtime::Runtime::new()?);

        let engine = runtime
            .block_on(async {
                SnowflakeEngine::new(
                    &user, &password, &account, &role, &warehouse, &database, &schema,
                )
                .await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyConnectionError, _>(e.to_string()))?;

        Ok(Self {
            engine: Arc::new(Mutex::new(engine)),
            runtime,
        })
    }

    /// Create a new SnowflakeAgent for AI completions
    fn create_agent(&self) -> PyResult<SnowflakeAgent> {
        let engine = Arc::clone(&self.engine);
        let runtime = Arc::clone(&self.runtime);

        let (account, database, schema, token) = runtime.block_on(async {
            let engine = engine.lock().await;
            let account = std::env::var("SNOWFLAKE_ACCOUNT").unwrap_or_default();
            let database = engine.get_database().await.unwrap_or_default();
            let schema = engine.get_schema().await.unwrap_or_default();
            let token = std::env::var("SNOWFLAKE_PASSWORD").unwrap_or_default();
            (account, database, schema, token)
        });

        let client = agents::AgentsClient::new(account, token, database, schema);

        Ok(SnowflakeAgent {
            client: Arc::new(tokio::sync::Mutex::new(client)),
            runtime,
        })
    }

    /// Get messages for a conversation
    fn get_messages(&self, conversation_id: String) -> PyResult<Vec<(String, String, String)>> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async {
            let engine = engine.lock().await;
            engine.get_messages(&conversation_id).await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Add a message to conversation history
    fn add_message(
        &self,
        conversation_id: String,
        message_id: String,
        role: String,
        content: String,
    ) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async {
            let engine = engine.lock().await;
            engine
                .add_message(conversation_id, message_id, role, content)
                .await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Get or create a conversation with settings
    fn get_or_create_conversation(
        &self,
        conversation_id: String,
        settings: String,
    ) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);
        let settings_json: serde_json::Value = serde_json::from_str(&settings)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let result = self.runtime.block_on(async {
            let engine = engine.lock().await;
            engine
                .get_or_create_conversation(&conversation_id, settings_json)
                .await
        });

        result
            .and_then(|v| serde_json::to_string(&v).map_err(|e| e.to_string()))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Update conversation settings
    fn update_conversation_settings(
        &self,
        conversation_id: String,
        settings: String,
    ) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        let settings_json: serde_json::Value = serde_json::from_str(&settings)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let result = self.runtime.block_on(async {
            let engine = engine.lock().await;
            engine
                .update_conversation_settings(&conversation_id, settings_json)
                .await
        });

        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    #[pyo3(signature = (prompt, model="openai-gpt-5-chat", temperature=0.7, max_tokens=4096, context=None, tools=None))]
    fn complete(
        &self,
        prompt: String,
        model: &str,
        temperature: f32,
        max_tokens: i32,
        context: Option<String>,
        tools: Option<Vec<agents::Tool>>,
    ) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);
        let runtime = Arc::clone(&self.runtime);
        let model = model.to_string();

        // Create an agent client and use it for completion
        let (account, database, schema, token) = runtime.block_on(async {
            let engine = engine.lock().await;
            let account = std::env::var("SNOWFLAKE_ACCOUNT").unwrap_or_default();
            let database = engine.get_database().await.unwrap_or_default();
            let schema = engine.get_schema().await.unwrap_or_default();
            let token = std::env::var("SNOWFLAKE_PASSWORD").unwrap_or_default();
            (account, database, schema, token)
        });

        let mut client = agents::AgentsClient::new(account, token, database, schema);

        // Add context as system message if provided
        if let Some(ctx) = context {
            client.add_system_message(&ctx);
        }

        let result = runtime.block_on(async {
            client
                .complete_with_tools(
                    &model,
                    &prompt,
                    false, // don't use history for single completion
                    Some(temperature),
                    None, // top_p
                    Some(max_tokens),
                    tools, // Pass tools to the completion
                    None,  // tool_choice
                )
                .await
        });

        match result {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    Ok(choice.message.get_content())
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "No response from model",
                    ))
                }
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    #[pyo3(signature = (messages, model="openai-gpt-5-chat", temperature=0.7, max_tokens=4096, context=None, tools=None))]
    fn chat(
        &self,
        messages: Vec<(String, String)>,
        model: &str,
        temperature: f32,
        max_tokens: i32,
        context: Option<String>,
        tools: Option<Vec<agents::Tool>>,
    ) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);
        let runtime = Arc::clone(&self.runtime);
        let model = model.to_string();

        let (account, database, schema, token) = runtime.block_on(async {
            let engine = engine.lock().await;
            let account = std::env::var("SNOWFLAKE_ACCOUNT").unwrap_or_default();
            let database = engine.get_database().await.unwrap_or_default();
            let schema = engine.get_schema().await.unwrap_or_default();
            let token = std::env::var("SNOWFLAKE_PASSWORD").unwrap_or_default();
            (account, database, schema, token)
        });

        let mut client = agents::AgentsClient::new(account, token, database, schema);

        // Add context as system message if provided
        if let Some(ctx) = context {
            client.add_system_message(&ctx);
        }

        // Convert messages to Message structs and add to history
        let mut formatted_messages: Vec<Message> = messages
            .iter()
            .map(|(role, content)| Message {
                role: Some(role.clone()),
                content: Some(content.clone()),
                content_list: None,
            })
            .collect();

        // Get the last user message for the completion
        let last_message = formatted_messages.pop().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("No messages provided")
        })?;

        let last_content = last_message.content.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Last message has no content")
        })?;

        // Add previous messages to conversation history
        for msg in formatted_messages {
            if let (Some(role), Some(content)) = (&msg.role, &msg.content) {
                match role.as_str() {
                    "user" | "human" => {
                        client
                            .conversation_history
                            .push(Message::new_user(content.clone()));
                    }
                    "assistant" | "ai" => {
                        client.add_assistant_response(content.clone());
                    }
                    "system" => {
                        client.add_system_message(content);
                    }
                    _ => {}
                }
            }
        }

        let result = runtime.block_on(async {
            client
                .complete_with_tools(
                    &model,
                    &last_content,
                    true, // use history
                    Some(temperature),
                    None, // top_p
                    Some(max_tokens),
                    tools,
                    None, // tool_choice
                )
                .await
        });

        match result {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    Ok(choice.message.get_content())
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "No response from model",
                    ))
                }
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    #[pyo3(signature = (messages, model="openai-gpt-5-chat", temperature=0.7, max_tokens=4096, context=None, tools=None))]
    fn chat_stream(
        &self,
        messages: Vec<(String, String)>,
        model: &str,
        temperature: f32,
        max_tokens: i32,
        context: Option<String>,
        tools: Option<Vec<agents::Tool>>,
    ) -> PyResult<PyStream> {
        let engine = Arc::clone(&self.engine);
        let runtime = Arc::clone(&self.runtime);
        let model = model.to_string();

        let (account, database, schema, token) = runtime.block_on(async {
            let engine = engine.lock().await;
            let account = std::env::var("SNOWFLAKE_ACCOUNT").unwrap_or_default();
            let database = engine.get_database().await.unwrap_or_default();
            let schema = engine.get_schema().await.unwrap_or_default();
            let token = std::env::var("SNOWFLAKE_PASSWORD").unwrap_or_default();
            (account, database, schema, token)
        });

        let mut client = agents::AgentsClient::new(account, token, database, schema);

        // Add context as system message if provided
        if let Some(ctx) = context {
            client.add_system_message(&ctx);
        }

        // Convert messages to Message structs
        let formatted_messages: Vec<Message> = messages
            .iter()
            .map(|(role, content)| Message {
                role: Some(role.clone()),
                content: Some(content.clone()),
                content_list: None,
            })
            .collect();

        let result = runtime.block_on(async {
            client
                .stream_complete(
                    &model,
                    formatted_messages,
                    Some(temperature),
                    None, // top_p
                    Some(max_tokens),
                    tools,
                )
                .await
        });

        match result {
            Ok((stream, _metadata)) => Ok(PyStream {
                stream: std::sync::Mutex::new(stream),
                runtime,
            }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// Execute SQL statement without result processing (e.g. DDL)
    fn execute_statement(&self, statement: String) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);

        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.execute_statement(&statement).await
        });

        result
            .and_then(|v| serde_json::to_string(&v).map_err(|e| e.to_string()))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Execute SQL query and return JSON result
    fn execute_query(&self, query: String) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);

        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.execute_query(&query, None, None, None).await
        });

        result
            .and_then(|v| serde_json::to_string(&v).map_err(|e| e.to_string()))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Validate SQL query
    fn validate_query(&self, query: String) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.validate_query(&query).await
        });

        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Get list of available AI models
    fn get_available_models(&self) -> PyResult<Vec<(String, String)>> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.get_available_models().await
        });

        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Close the connection
    fn close(&self) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let mut engine = engine.lock().await;
            engine.close().await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// List all accessible databases
    fn list_databases(&self) -> PyResult<Vec<String>> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.list_databases().await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// List all schemas in a database (or current if None)
    #[pyo3(signature = (database = None))]
    fn list_schemas(&self, database: Option<String>) -> PyResult<Vec<String>> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.list_schemas(database.as_deref()).await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// List all accessible warehouses
    fn list_warehouses(&self) -> PyResult<Vec<String>> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.list_warehouses().await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Switch to a different database
    fn use_database(&self, database: String) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let mut engine = engine.lock().await;
            engine.use_database(&database).await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Switch to a different schema
    fn use_schema(&self, schema: String) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let mut engine = engine.lock().await;
            engine.use_schema(&schema).await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Switch to a different warehouse
    fn use_warehouse(&self, warehouse: String) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let mut engine = engine.lock().await;
            engine.use_warehouse(&warehouse).await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Switch database and schema together
    fn use_database_schema(&self, database: String, schema: String) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let mut engine = engine.lock().await;
            engine.use_database_schema(&database, &schema).await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// List all tables across all accessible databases and schemas
    fn list_all_tables(&self) -> PyResult<Vec<(String, String, String)>> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.list_all_tables().await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// List tables in a specific database and schema
    fn list_tables_in(&self, database: String, schema: String) -> PyResult<Vec<String>> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.list_tables_in(&database, &schema).await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Get current database
    fn get_current_database(&self) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            // Use the getter method
            engine.get_database().await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Get current schema
    fn get_current_schema(&self) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.get_schema().await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Get current user
    fn get_current_user(&self) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            // Use the getter method
            engine.get_user().await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Get sample data for a specific table, limited to 1 row.
    fn get_table_sample_data_rust(&self, table_name: String) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);

        let result = self.runtime.block_on(async move {
            let mut engine = engine.lock().await;

            let parts: Vec<&str> = table_name.split('.').collect();
            let (database, schema, table) = match parts.len() {
                3 => (Some(parts[0]), Some(parts[1]), parts[2]),
                2 => (Some(parts[0]), None, parts[1]),
                1 => (None, None, parts[0]),
                _ => return Err("Invalid table name format".to_string()),
            };

            if let (Some(db), Some(sch)) = (database, schema) {
                if !db.is_empty() && !sch.is_empty() {
                    engine
                        .use_database_schema(db, sch)
                        .await
                        .map_err(|e| e.to_string())?;
                }
            } else if let Some(db) = database {
                if !db.is_empty() {
                    engine.use_database(db).await.map_err(|e| e.to_string())?;
                }
            } else if let Some(sch) = schema {
                if !sch.is_empty() {
                    engine.use_schema(sch).await.map_err(|e| e.to_string())?;
                }
            }

            let query = format!("SELECT * FROM {}", table);
            engine.execute_query(&query, Some(0), Some(1), None).await
        });

        result
            .and_then(|v| serde_json::to_string(&v).map_err(|e| e.to_string()))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    fn get_table_info(&self, table_name: String) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.get_table_info(&table_name).await
        });

        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Get tool definition for get_multiple_table_definitions
    fn get_multiple_table_definitions_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_multiple_table_definitions",
                "description": "Get schema definitions for multiple tables at once.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_names": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of fully qualified table names (e.g., DATABASE.SCHEMA.TABLE or just TABLE)"
                        }
                    },
                    "required": ["table_names"]
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for list_tables_in
    fn list_tables_in_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "list_tables_in",
                "description": "List tables in a specific database and schema.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "Database name"
                        },
                        "schema": {
                            "type": "string",
                            "description": "Schema name"
                        }
                    },
                    "required": ["database", "schema"]
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for validate_query
    fn validate_query_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "validate_query",
                "description": "Validate a SQL query without executing it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The SQL query to validate"
                        }
                    },
                    "required": ["query"]
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for extract_answer (uses AI_EXTRACT SQL function)
    fn extract_answer_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "extract_answer",
                "description": "Extract specific data points from a document using AI_EXTRACT. ONLY use for discrete facts like 'What is the salary?', 'What is the date?', 'How many shares?'. DO NOT use for summaries, explanations, or open-ended questions - use semantic search via execute_query on DOCUMENT_EMBEDDINGS for those instead.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The stage path to the document (e.g., '@DATABASE.SCHEMA.STAGE/file.pdf')"
                        },
                        "questions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of SHORT, SPECIFIC questions to extract discrete answers for (e.g., 'What is the base salary?', 'What is the start date?'). NOT for summaries or explanations."
                        }
                    },
                    "required": ["file_path", "questions"]
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for sentiment
    fn sentiment_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "sentiment",
                "description": "Analyze the sentiment of a text using Cortex.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to analyze"
                        }
                    },
                    "required": ["text"]
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for summarize
    fn summarize_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "summarize",
                "description": "Summarize a text using Cortex.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to summarize"
                        }
                    },
                    "required": ["text"]
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for translate
    fn translate_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "translate",
                "description": "Translate text from one language to another using Cortex.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to translate"
                        },
                        "to_lang": {
                            "type": "string",
                            "description": "Target language code (e.g., 'en', 'fr')"
                        },
                        "from_lang": {
                            "type": "string",
                            "description": "Source language code (optional)"
                        }
                    },
                    "required": ["text", "to_lang"]
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for get_table_sample_data
    fn get_table_sample_data_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_table_sample_data",
                "description": "Get sample data from a Snowflake table. Returns up to 1 row of data to understand the table structure and content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "The fully qualified table name (e.g., DATABASE.SCHEMA.TABLE or just TABLE)"
                        }
                    },
                    "required": ["table_name"]
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for execute_query
    fn execute_query_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "execute_query",
                "description": "Execute a SQL query against Snowflake and return the results as JSON.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The SQL query to execute"
                        }
                    },
                    "required": ["query"]
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for execute_statement
    fn execute_statement_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "execute_statement",
                "description": "Execute any Snowflake SQL statement (DDL/DML/SHOW) and return structured results when available.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": "The exact SQL statement to run"
                        }
                    },
                    "required": ["statement"]
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for list_tables
    fn list_tables_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "list_tables",
                "description": "List all tables accessible in the current Snowflake context.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for get_table_schema
    fn get_table_schema_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_table_schema",
                "description": "Get the schema/structure of a Snowflake table including column names, types, and constraints.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "The fully qualified table name (e.g., DATABASE.SCHEMA.TABLE or just TABLE)"
                        }
                    },
                    "required": ["table_name"]
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for list_databases
    fn list_databases_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "list_databases",
                "description": "List all databases accessible in Snowflake.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for list_schemas
    fn list_schemas_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "list_schemas",
                "description": "List all schemas in a database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "The database name. If not provided, uses the current database."
                        }
                    },
                    "required": []
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get tool definition for describe_table
    fn describe_table_tool(&self) -> PyResult<String> {
        let tool_def = serde_json::json!({
            "type": "function",
            "function": {
                "name": "describe_table",
                "description": "Describe a Snowflake table structure using DESCRIBE TABLE command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "The fully qualified table name (e.g., DATABASE.SCHEMA.TABLE or just TABLE)"
                        }
                    },
                    "required": ["table_name"]
                }
            }
        });
        serde_json::to_string(&tool_def)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// List all stages accessible to the user
    fn list_stages(&self) -> PyResult<Vec<String>> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.list_stages().await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// List documents available in Cortex (from DOCUMENT_PARSE_RESULTS table)
    /// Returns a list of tuples with (file_name, stage_path, is_parsed, page_count)
    fn list_cortex_documents(&self) -> PyResult<Vec<(String, String, bool, i64)>> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.list_cortex_documents().await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Download a document from Snowflake stage and return base64-encoded content
    /// document_id is the FILE_NAME from DOCUMENT_PARSE_RESULTS
    fn download_cortex_document(&self, document_id: String) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.download_cortex_document(&document_id).await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Upload bytes directly to a Snowflake stage (for widget PDF uploads)
    /// Returns the stage path where the file was uploaded
    #[pyo3(signature = (file_bytes, file_name, stage_name = None))]
    fn upload_bytes_to_stage(
        &self,
        file_bytes: Vec<u8>,
        file_name: String,
        stage_name: Option<String>,
    ) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine
                .upload_bytes_to_stage(&file_bytes, &file_name, stage_name.as_deref())
                .await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// List files in a specific stage
    #[pyo3(signature = (stage_name, no_cache = false))]
    fn list_files_in_stage(&self, stage_name: String, no_cache: bool) -> PyResult<Vec<String>> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let mut engine = engine.lock().await;

            let parts: Vec<&str> = stage_name.split('.').collect();
            let (database, schema, stage) = match parts.len() {
                3 => (Some(parts[0]), Some(parts[1]), parts[2]),
                2 => (None, Some(parts[0]), parts[1]),
                1 => (None, None, parts[0]),
                _ => return Err("Invalid stage name format".to_string()),
            };

            if let (Some(db), Some(sch)) = (database, schema) {
                if !db.is_empty() && !sch.is_empty() {
                    engine
                        .use_database_schema(db, sch)
                        .await
                        .map_err(|e| e.to_string())?;
                }
            } else if let Some(sch) = schema {
                if !sch.is_empty() {
                    engine.use_schema(sch).await.map_err(|e| e.to_string())?;
                }
            }

            engine.list_files_in_stage(stage, no_cache).await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Upload a file to a Snowflake stage.
    #[pyo3(signature = (file_path, stage_name))]
    fn upload_file_to_stage(&self, file_path: String, stage_name: String) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine
                .upload_file_to_stage(std::path::Path::new(&file_path), &stage_name)
                .await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Parse a document from a Snowflake stage.
    #[pyo3(signature = (stage_file_path, mode=None))]
    fn parse_document(&self, stage_file_path: String, mode: Option<String>) -> PyResult<String> {
        let engine = self.engine.clone();
        let mode_ref = mode.as_deref();

        self.runtime.block_on(async move {
            let engine_lock = engine.lock().await;
            engine_lock
                .ai_parse_document(&stage_file_path, mode_ref)
                .await
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
        })
    }

    /// Set conversation-specific data in Snowflake tables
    #[pyo3(signature = (conversation_id, key, value))]
    fn set_conversation_data(
        &self,
        conversation_id: String,
        key: String,
        value: String,
    ) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async {
            let engine = engine.lock().await;
            let user = engine.get_user().await.map_err(|e| e.to_string())?;
            let schema_name = format!("USER_{}", user.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect::<String>().to_uppercase());
            let table_name = format!("OPENBB_AGENTS.{}.AGENTS_CONTEXT_OBJECTS", schema_name);

            // Use MERGE to insert or update
            let object_id = format!("{}_{}", conversation_id, key);
            let escaped_object_id = object_id.replace("'", "''");
            let escaped_conv_id = conversation_id.replace("'", "''");
            let escaped_key = key.replace("'", "''");

            // Check if value is already valid JSON - if so, store it directly
            // If not, wrap it in {"value": "..."} 
            let json_value = if serde_json::from_str::<serde_json::Value>(&value).is_ok() {
                // Value is already valid JSON - use it directly
                // Just need to escape $$ delimiter if present
                value.replace("$$", "$ $")
            } else {
                // Value is a plain string - wrap it in JSON object
                let escaped = value
                    .replace("\\", "\\\\")
                    .replace("\"", "\\\"")
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t")
                    .replace("$$", "$ $");
                format!("{{\"value\": \"{}\"}}", escaped)
            };

            let query = format!(
                "MERGE INTO {} AS target
                 USING (SELECT '{}' AS OBJECT_ID) AS source
                 ON target.OBJECT_ID = source.OBJECT_ID
                 WHEN MATCHED THEN UPDATE SET OBJECT_VALUE = PARSE_JSON($${}$$)
                 WHEN NOT MATCHED THEN INSERT (OBJECT_ID, CONVERSATION_ID, OBJECT_TYPE, OBJECT_KEY, OBJECT_VALUE)
                 VALUES ('{}', '{}', 'context', '{}', PARSE_JSON($${}$$))",
                table_name, escaped_object_id, json_value, escaped_object_id, escaped_conv_id, escaped_key, json_value
            );
            engine.execute_statement(&query).await.map(|_| ())
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Get conversation-specific data from Snowflake tables
    #[pyo3(signature = (conversation_id, key))]
    fn get_conversation_data(
        &self,
        conversation_id: String,
        key: String,
    ) -> PyResult<Option<String>> {
        let engine = Arc::clone(&self.engine);
        let debug_enabled = std::env::var("SNOWFLAKE_DEBUG").is_ok();
        let result: Result<Option<String>, String> = self.runtime.block_on(async {
            let engine = engine.lock().await;
            let user = engine.get_user().await.map_err(|e| e.to_string())?;
            let schema_name = format!("USER_{}", user.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect::<String>().to_uppercase());
            let table_name = format!("OPENBB_AGENTS.{}.AGENTS_CONTEXT_OBJECTS", schema_name);

            let object_id = format!("{}_{}", conversation_id, key);
            // Try to get either the wrapped value or the raw JSON
            // COALESCE handles both formats: {"value": "..."} and raw JSON objects
            let query = format!(
                "SELECT COALESCE(OBJECT_VALUE:value::STRING, TO_JSON(OBJECT_VALUE)::STRING) AS value FROM {} WHERE OBJECT_ID = '{}'",
                table_name, object_id
            );

            if debug_enabled {
                eprintln!("[DEBUG] get_conversation_data query: {}", query);
            }

            let result = engine.execute_statement(&query).await?;

            if debug_enabled {
                eprintln!("[DEBUG] get_conversation_data result: {:?}", result);
            }

            // Parse the JSON result
            if let Some(rows) = result.as_array() {
                if let Some(first_row) = rows.first() {
                    if debug_enabled {
                        eprintln!("[DEBUG] get_conversation_data first_row: {:?}", first_row);
                    }
                    if let Some(value) = first_row.get("VALUE").or(first_row.get("value")) {
                        if debug_enabled {
                            eprintln!("[DEBUG] get_conversation_data value: {:?}", value);
                        }
                        if let Some(s) = value.as_str() {
                            return Ok(Some(s.to_string()));
                        }
                        // Also try to handle if value is a JSON object/number and convert to string
                        if !value.is_null() {
                            return Ok(Some(value.to_string()));
                        }
                    }
                }
            }
            Ok(None)
        });
        result.map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Delete conversation-specific data from Snowflake tables
    #[pyo3(signature = (conversation_id, key))]
    fn delete_conversation_data(&self, conversation_id: String, key: String) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async {
            let engine = engine.lock().await;
            let user = engine.get_user().await.map_err(|e| e.to_string())?;
            let schema_name = format!(
                "USER_{}",
                user.chars()
                    .map(|c| if c.is_alphanumeric() { c } else { '_' })
                    .collect::<String>()
                    .to_uppercase()
            );
            let table_name = format!("OPENBB_AGENTS.{}.AGENTS_CONTEXT_OBJECTS", schema_name);

            let object_id = format!("{}_{}", conversation_id, key);
            let query = format!(
                "DELETE FROM {} WHERE OBJECT_ID = '{}'",
                table_name, object_id
            );
            engine.execute_statement(&query).await.map(|_| ())
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// List all conversation-specific data for a conversation
    #[pyo3(signature = (conversation_id))]
    fn list_conversation_data(&self, conversation_id: String) -> PyResult<Vec<(String, String)>> {
        let engine = Arc::clone(&self.engine);
        let result: Result<Vec<(String, String)>, String> = self.runtime.block_on(async {
            let engine = engine.lock().await;
            let user = engine.get_user().await.map_err(|e| e.to_string())?;
            let schema_name = format!("USER_{}", user.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect::<String>().to_uppercase());
            let table_name = format!("OPENBB_AGENTS.{}.AGENTS_CONTEXT_OBJECTS", schema_name);

            let query = format!(
                "SELECT OBJECT_KEY, COALESCE(OBJECT_VALUE:value::STRING, TO_JSON(OBJECT_VALUE)::STRING) AS value FROM {} WHERE CONVERSATION_ID = '{}'",
                table_name, conversation_id
            );
            let result = engine.execute_statement(&query).await?;

            let mut data = Vec::new();
            if let Some(rows) = result.as_array() {
                for row in rows {
                    let key = row.get("OBJECT_KEY")
                        .or(row.get("object_key"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let value = row.get("VALUE")
                        .or(row.get("value"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    if !key.is_empty() {
                        data.push((key, value));
                    }
                }
            }
            Ok(data)
        });
        result.map_err(|e: String| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Clear conversation (delete all messages)
    #[pyo3(signature = (conversation_id))]
    fn clear_conversation(&self, conversation_id: String) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        self.runtime
            .block_on(async move {
                let engine = engine.lock().await;
                // Clear from Snowflake tables - use getter method for user
                let user = engine.get_user().await.map_err(|e| e.to_string())?;
                let schema_name = format!(
                    "USER_{}",
                    user.chars()
                        .map(|c| if c.is_alphanumeric() { c } else { '_' })
                        .collect::<String>()
                        .to_uppercase()
                );
                let table_name = format!("OPENBB_AGENTS.{}.AGENTS_MESSAGES", schema_name);
                let query = format!(
                    "DELETE FROM {} WHERE CONVERSATION_ID = '{}'",
                    table_name, conversation_id
                );
                engine.execute_statement(&query).await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }
}

#[pyclass]
struct SnowflakeAgent {
    client: Arc<tokio::sync::Mutex<agents::AgentsClient>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pyclass]
struct PyStream {
    stream:
        std::sync::Mutex<Pin<Box<dyn futures::Stream<Item = Result<StreamChunk, String>> + Send>>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self) -> PyResult<Option<agents::StreamChunk>> {
        let mut stream = self.stream.lock().unwrap();
        let result = self.runtime.block_on(stream.next());
        match result {
            Some(Ok(chunk)) => Ok(Some(chunk)),
            Some(Err(e)) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
            None => Ok(None),
        }
    }
}

#[pymethods]
impl SnowflakeAgent {
    #[pyo3(signature = (message, model="openai-gpt-5-chat", temperature=None, max_tokens=None))]
    fn stream_complete(
        &self,
        message: String,
        model: &str,
        temperature: Option<f32>,
        max_tokens: Option<i32>,
    ) -> PyResult<PyStream> {
        let client = Arc::clone(&self.client);
        let model = model.to_string();
        let runtime = Arc::clone(&self.runtime);

        let result = self.runtime.block_on(async move {
            let mut client = client.lock().await;
            client
                .stream_complete(
                    &model,
                    vec![Message::new_user(message.to_string())],
                    temperature,
                    None, // top_p
                    max_tokens,
                    None, // tools
                )
                .await
        });

        match result {
            Ok((stream, _metadata)) => Ok(PyStream {
                stream: std::sync::Mutex::new(stream),
                runtime,
            }),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    #[pyo3(signature = (message, model="openai-gpt-5-chat", temperature=None, max_tokens=None, use_history=true))]
    fn complete(
        &self,
        message: String,
        model: &str,
        temperature: Option<f32>,
        max_tokens: Option<i32>,
        use_history: bool,
    ) -> PyResult<String> {
        let client = Arc::clone(&self.client);
        let model = model.to_string();

        let result = self.runtime.block_on(async move {
            let mut client = client.lock().await;
            client
                .complete(&model, &message, use_history, temperature, None, max_tokens)
                .await
        });

        match result {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    Ok(choice.message.get_content())
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "No response from model",
                    ))
                }
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    #[pyo3(signature = (message, model="openai-gpt-5-chat", temperature=None, max_tokens=None, use_history=true, tools=None, tool_choice=None))]
    fn complete_with_tools(
        &self,
        message: String,
        model: &str,
        temperature: Option<f32>,
        max_tokens: Option<i32>,
        use_history: bool,
        tools: Option<Vec<agents::Tool>>,
        tool_choice: Option<String>,
    ) -> PyResult<String> {
        let client = Arc::clone(&self.client);
        let model = model.to_string();

        let rust_tool_choice = tool_choice.map(|tc_str| match tc_str.as_str() {
            "auto" => agents::ToolChoice::Auto("auto".to_string()),
            "required" => agents::ToolChoice::Required("required".to_string()),
            "none" => agents::ToolChoice::None("none".to_string()),
            _ => agents::ToolChoice::Specific {
                tool_type: "function".to_string(),
                function: agents::FunctionName { name: tc_str },
            },
        });

        let result = self.runtime.block_on(async move {
            let mut client = client.lock().await;
            client
                .complete_with_tools(
                    &model,
                    &message,
                    use_history,
                    temperature,
                    None,
                    max_tokens,
                    tools,
                    rust_tool_choice,
                )
                .await
        });

        match result {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    Ok(choice.message.get_content())
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "No response from model",
                    ))
                }
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    #[pyo3(signature = (message, model="openai-gpt-5-chat", temperature=None, max_tokens=None, use_history=true))]
    fn get_complete_details(
        &self,
        message: String,
        model: &str,
        temperature: Option<f32>,
        max_tokens: Option<i32>,
        use_history: bool,
    ) -> PyResult<(String, i32, String, i32, i32, i32)> {
        let client = Arc::clone(&self.client);
        let model = model.to_string();

        let result = self.runtime.block_on(async move {
            let mut client = client.lock().await;
            client
                .complete(&model, &message, use_history, temperature, None, max_tokens)
                .await
        });

        match result {
            Ok(response) => {
                if let Some(choice) = response.choices.first() {
                    let finish_reason = choice
                        .finish_reason
                        .as_ref()
                        .map(|s| s.as_str())
                        .unwrap_or("unknown");

                    Ok((
                        choice.message.get_content(),
                        choice.index,
                        finish_reason.to_string(),
                        response.usage.prompt_tokens,
                        response.usage.completion_tokens,
                        response.usage.total_tokens,
                    ))
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "No response from model",
                    ))
                }
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    fn reset_conversation(&self) -> PyResult<()> {
        let client = Arc::clone(&self.client);
        self.runtime.block_on(async move {
            let mut client = client.lock().await;
            client.reset_conversation();
        });
        Ok(())
    }

    fn get_conversation_history(&self) -> PyResult<Vec<(String, String)>> {
        let client = Arc::clone(&self.client);
        let history = self.runtime.block_on(async move {
            let client = client.lock().await;
            client
                .get_conversation_history()
                .iter()
                .map(|msg| {
                    let role = msg
                        .role
                        .as_ref()
                        .map(|s| s.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let content = msg.get_content();
                    (role, content)
                })
                .collect::<Vec<_>>()
        });
        Ok(history)
    }

    fn get_usage_stats(&self) -> PyResult<(usize, i64, i64, i64)> {
        let client = Arc::clone(&self.client);
        let stats = self.runtime.block_on(async move {
            let client = client.lock().await;
            let stats = client.get_usage_stats();
            (
                stats.total_requests,
                stats.total_prompt_tokens,
                stats.total_completion_tokens,
                stats.total_tokens,
            )
        });
        Ok(stats)
    }

    fn reset_usage_stats(&self) -> PyResult<()> {
        let client = Arc::clone(&self.client);
        self.runtime.block_on(async move {
            let mut client = client.lock().await;
            client.reset_usage_stats();
        });
        Ok(())
    }

    fn get_last_metadata(&self) -> PyResult<Option<(String, String, i64, String)>> {
        let client = Arc::clone(&self.client);
        let metadata = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.get_last_metadata().map(|m| {
                (
                    m.request_id.clone(),
                    m.model.clone(),
                    m.created,
                    m.object.clone(),
                )
            })
        });
        Ok(metadata)
    }

    fn get_usage_report(&self) -> PyResult<String> {
        let client = Arc::clone(&self.client);
        let report = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.get_usage_report()
        });
        Ok(report)
    }

    /// Create a new agent
    #[pyo3(signature = (name, model, instructions, description=None, temperature=None, top_p=None, max_tokens=None, tools=None, create_mode=None, system_instructions=None))]
    fn create_agent(
        &self,
        name: String,
        model: String,
        instructions: String,
        description: Option<String>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<i32>,
        tools: Option<Vec<(String, String, String)>>,
        create_mode: Option<String>,
        system_instructions: Option<String>,
    ) -> PyResult<String> {
        let client = Arc::clone(&self.client);

        let tools_vec = tools.map(|t| {
            t.into_iter()
                .map(|(name, desc, params_json)| {
                    let params: serde_json::Value =
                        serde_json::from_str(&params_json).unwrap_or(serde_json::json!({}));

                    // Convert JSON value to ToolInputSchema
                    let input_schema = agents::ToolInputSchema {
                        schema_type: params
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("object")
                            .to_string(),
                        description: params
                            .get("description")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        properties: params.get("properties").and_then(|v| v.as_object()).map(
                            |props| {
                                props
                                    .iter()
                                    .map(|(k, v)| {
                                        (
                                            k.clone(),
                                            agents::ToolInputSchema {
                                                schema_type: v
                                                    .get("type")
                                                    .and_then(|t| t.as_str())
                                                    .unwrap_or("string")
                                                    .to_string(),
                                                description: v
                                                    .get("description")
                                                    .and_then(|d| d.as_str())
                                                    .map(|s| s.to_string()),
                                                properties: None,
                                                items: None,
                                                required: None,
                                            },
                                        )
                                    })
                                    .collect()
                            },
                        ),
                        items: None,
                        required: params
                            .get("required")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect()
                            }),
                    };

                    agents::Tool {
                        tool_spec: agents::ToolSpec {
                            tool_type: "generic".to_string(),
                            name,
                            description: desc,
                            input_schema,
                        },
                    }
                })
                .collect()
        });

        let request = agents::CreateAgentRequest {
            name,
            comment: description,
            profile: None,
            models: agents::ModelConfig {
                model,
                temperature,
                top_p,
                max_tokens,
            },
            instructions: agents::AgentInstructions {
                response: instructions,
                orchestration: None,
                system: system_instructions,
                sample_questions: None,
            },
            orchestration: None,
            tools: tools_vec,
            tool_resources: None,
        };

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.create_agent(request, create_mode.as_deref()).await
        });

        match result {
            Ok(agent) => {
                let json = serde_json::json!({
                    "id": agent.id,
                    "name": agent.name,
                    "model": agent.models.model,
                    "comment": agent.comment,
                    "instructions": agent.instructions.response,
                    "temperature": agent.models.temperature,
                    "top_p": agent.models.top_p,
                    "max_tokens": agent.models.max_tokens,
                    "created_on": agent.created_on,
                    "updated_on": agent.updated_on,
                });
                serde_json::to_string(&json)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// Create agent from JSON file
    #[pyo3(signature = (file_path, create_mode=None))]
    fn create_agent_from_file(
        &self,
        file_path: String,
        create_mode: Option<String>,
    ) -> PyResult<String> {
        let client = Arc::clone(&self.client);

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client
                .create_agent_from_file(&file_path, create_mode.as_deref())
                .await
        });

        match result {
            Ok(agent) => {
                let json = serde_json::json!({
                    "id": agent.id,
                    "name": agent.name,
                    "model": agent.models.model,
                    "comment": agent.comment,
                    "instructions": agent.instructions.response,
                    "temperature": agent.models.temperature,
                    "top_p": agent.models.top_p,
                    "max_tokens": agent.models.max_tokens,
                    "tools": agent.tools,
                    "created_on": agent.created_on,
                    "updated_on": agent.updated_on,
                });
                serde_json::to_string(&json)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// Get an agent by name
    fn get_agent(&self, agent_name: String) -> PyResult<String> {
        let client = Arc::clone(&self.client);

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.get_agent(&agent_name).await
        });

        match result {
            Ok(agent) => {
                let json = serde_json::json!({
                    "id": agent.id,
                    "name": agent.name,
                    "model": agent.models.model,
                    "comment": agent.comment,
                    "instructions": agent.instructions.response,
                    "temperature": agent.models.temperature,
                    "top_p": agent.models.top_p,
                    "max_tokens": agent.models.max_tokens,
                    "tools": agent.tools,
                    "created_on": agent.created_on,
                    "updated_on": agent.updated_on,
                });
                serde_json::to_string(&json)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// List all agents
    #[pyo3(signature = (like=None, from_name=None, show_limit=None))]
    fn list_agents(
        &self,
        like: Option<String>,
        from_name: Option<String>,
        show_limit: Option<i32>,
    ) -> PyResult<String> {
        let client = Arc::clone(&self.client);

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client
                .list_agents(like.as_deref(), from_name.as_deref(), show_limit)
                .await
        });

        match result {
            Ok(agents) => {
                let json_agents: Vec<_> = agents
                    .into_iter()
                    .map(|agent| {
                        serde_json::json!({
                            "id": agent.id,
                            "name": agent.name,
                            "model": agent.models.model,
                            "comment": agent.comment,
                            "instructions": agent.instructions.response,
                            "temperature": agent.models.temperature,
                            "top_p": agent.models.top_p,
                            "max_tokens": agent.models.max_tokens,
                            "created_on": agent.created_on,
                            "updated_on": agent.updated_on,
                        })
                    })
                    .collect();
                serde_json::to_string(&json_agents)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// Update an agent
    #[pyo3(signature = (agent_name, name=None, model=None, instructions=None, description=None, temperature=None, top_p=None, max_tokens=None, tools=None, system_instructions=None))]
    fn update_agent(
        &self,
        agent_name: String,
        name: Option<String>,
        model: Option<String>,
        instructions: Option<String>,
        description: Option<String>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<i32>,
        tools: Option<Vec<(String, String, String)>>,
        system_instructions: Option<String>,
    ) -> PyResult<String> {
        let client = Arc::clone(&self.client);

        let tools_vec = tools.map(|t| {
            t.into_iter()
                .map(|(name, desc, params_json)| {
                    let params: serde_json::Value =
                        serde_json::from_str(&params_json).unwrap_or(serde_json::json!({}));

                    let input_schema = agents::ToolInputSchema {
                        schema_type: params
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("object")
                            .to_string(),
                        description: params
                            .get("description")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        properties: params.get("properties").and_then(|v| v.as_object()).map(
                            |props| {
                                props
                                    .iter()
                                    .map(|(k, v)| {
                                        (
                                            k.clone(),
                                            agents::ToolInputSchema {
                                                schema_type: v
                                                    .get("type")
                                                    .and_then(|t| t.as_str())
                                                    .unwrap_or("string")
                                                    .to_string(),
                                                description: v
                                                    .get("description")
                                                    .and_then(|d| d.as_str())
                                                    .map(|s| s.to_string()),
                                                properties: None,
                                                items: None,
                                                required: None,
                                            },
                                        )
                                    })
                                    .collect()
                            },
                        ),
                        items: None,
                        required: params
                            .get("required")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                    .collect()
                            }),
                    };

                    agents::Tool {
                        tool_spec: agents::ToolSpec {
                            tool_type: "generic".to_string(),
                            name,
                            description: desc,
                            input_schema,
                        },
                    }
                })
                .collect()
        });

        let models = if model.is_some()
            || temperature.is_some()
            || top_p.is_some()
            || max_tokens.is_some()
        {
            Some(agents::ModelConfig {
                model: model.unwrap_or_default(),
                temperature,
                top_p,
                max_tokens,
            })
        } else {
            None
        };

        let instructions_config = if instructions.is_some() || system_instructions.is_some() {
            Some(agents::AgentInstructions {
                response: instructions.unwrap_or_default(),
                orchestration: None,
                system: system_instructions,
                sample_questions: None,
            })
        } else {
            None
        };

        let request = agents::UpdateAgentRequest {
            name,
            comment: description,
            profile: None,
            models,
            instructions: instructions_config,
            orchestration: None,
            tools: tools_vec,
            tool_resources: None,
        };

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.update_agent(&agent_name, request).await
        });

        match result {
            Ok(agent) => {
                let json = serde_json::json!({
                    "id": agent.id,
                    "name": agent.name,
                    "model": agent.models.model,
                    "comment": agent.comment,
                    "instructions": agent.instructions.response,
                    "temperature": agent.models.temperature,
                    "top_p": agent.models.top_p,
                    "max_tokens": agent.models.max_tokens,
                    "created_on": agent.created_on,
                    "updated_on": agent.updated_on,
                });
                serde_json::to_string(&json)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// Delete an agent
    #[pyo3(signature = (agent_name, if_exists=None))]
    fn delete_agent(&self, agent_name: String, if_exists: Option<bool>) -> PyResult<()> {
        let client = Arc::clone(&self.client);

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.delete_agent(&agent_name, if_exists).await
        });

        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Create a session for an agent
    #[pyo3(signature = (agent_id, metadata=None))]
    fn create_session(&self, agent_id: String, metadata: Option<String>) -> PyResult<String> {
        let client = Arc::clone(&self.client);

        let metadata_json = metadata.and_then(|m| serde_json::from_str(&m).ok());

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.create_session(&agent_id, metadata_json).await
        });

        match result {
            Ok(session) => {
                let json = serde_json::json!({
                    "id": session.id,
                    "agent_id": session.agent_id,
                    "created_on": session.created_on,
                    "updated_on": session.updated_on,
                    "metadata": session.metadata,
                });
                serde_json::to_string(&json)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// Get a session by ID
    fn get_session(&self, session_id: String) -> PyResult<String> {
        let client = Arc::clone(&self.client);

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.get_session(&session_id).await
        });

        match result {
            Ok(session) => {
                let json = serde_json::json!({
                    "id": session.id,
                    "agent_id": session.agent_id,
                    "created_on": session.created_on,
                    "updated_on": session.updated_on,
                    "metadata": session.metadata,
                });
                serde_json::to_string(&json)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// List all sessions for an agent
    fn list_sessions(&self, agent_id: String) -> PyResult<String> {
        let client = Arc::clone(&self.client);

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.list_sessions(&agent_id).await
        });

        match result {
            Ok(sessions) => {
                let json_sessions: Vec<_> = sessions
                    .into_iter()
                    .map(|s| {
                        serde_json::json!({
                            "id": s.id,
                            "agent_id": s.agent_id,
                            "created_on": s.created_on,
                            "updated_on": s.updated_on,
                            "metadata": s.metadata,
                        })
                    })
                    .collect();
                serde_json::to_string(&json_sessions)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// Delete a session
    fn delete_session(&self, session_id: String) -> PyResult<()> {
        let client = Arc::clone(&self.client);

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.delete_session(&session_id).await
        });

        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Send a message to a session (non-streaming)
    fn send_message(&self, session_id: String, content: String) -> PyResult<String> {
        let client = Arc::clone(&self.client);

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.send_message(&session_id, &content).await
        });

        match result {
            Ok(response) => {
                let json = serde_json::json!({
                    "id": response.id,
                    "session_id": response.session_id,
                    "role": response.role,
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                    "created_on": response.created_on,
                    "usage": response.usage,
                });
                serde_json::to_string(&json)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// List messages in a session
    fn list_messages(&self, session_id: String) -> PyResult<String> {
        let client = Arc::clone(&self.client);

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.list_messages(&session_id).await
        });

        match result {
            Ok(messages) => {
                let json_messages: Vec<_> = messages
                    .into_iter()
                    .map(|m| {
                        serde_json::json!({
                            "id": m.id,
                            "session_id": m.session_id,
                            "role": m.role,
                            "content": m.content,
                            "created_on": m.created_on,
                        })
                    })
                    .collect();
                serde_json::to_string(&json_messages)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }

    /// Get a specific message
    fn get_message(&self, session_id: String, message_id: String) -> PyResult<String> {
        let client = Arc::clone(&self.client);

        let result = self.runtime.block_on(async move {
            let client = client.lock().await;
            client.get_message(&session_id, &message_id).await
        });

        match result {
            Ok(message) => {
                let json = serde_json::json!({
                    "id": message.id,
                    "session_id": message.session_id,
                    "role": message.role,
                    "content": message.content,
                    "created_on": message.created_on,
                });
                serde_json::to_string(&json)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            }
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e)),
        }
    }
}

#[pymodule]
fn _snowflake_ai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SnowflakeAI>()?;
    m.add_class::<SnowflakeAgent>()?;
    m.add_class::<PyStream>()?;
    m.add_class::<agents::Message>()?;
    m.add_class::<agents::SnowflakeFile>()?;
    m.add_class::<agents::DocumentUrl>()?;
    m.add_class::<agents::ContentItem>()?;
    m.add_class::<agents::Tool>()?;
    m.add_class::<agents::ToolSpec>()?;
    m.add_class::<agents::ToolInputSchema>()?;
    m.add_class::<agents::ToolCall>()?;
    m.add_class::<agents::FunctionCall>()?;
    m.add_class::<agents::StreamChunk>()?;
    m.add_class::<agents::StreamChoice>()?;
    m.add_class::<agents::Delta>()?;
    m.add_class::<agents::Usage>()?;
    m.add_class::<agents::CompleteResponse>()?;
    m.add_class::<agents::Choice>()?;
    m.add_class::<agents::FunctionName>()?;
    Ok(())
}
