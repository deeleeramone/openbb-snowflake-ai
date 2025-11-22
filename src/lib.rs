#![cfg(feature = "extension-module")]

pub mod agents;
pub mod cache;
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

    #[pyo3(signature = (prompt, model="llama3.1-70b", temperature=0.7, max_tokens=4096, context=None, tools=None))]
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
        let model = model.to_string();
        let full_prompt = if let Some(ctx) = context {
            format!("{}\n\n{}", ctx, prompt)
        } else {
            prompt
        };

        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;

            let options = serde_json::json!({
                "temperature": temperature,
                "max_tokens": max_tokens,
            });

            engine
                .ai_complete(&model, &full_prompt, Some(options), tools)
                .await
        });

        result
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
            .map(|s| s.replace("\\_", "_"))
    }

    #[pyo3(signature = (messages, model="llama3.1-70b", temperature=0.7, max_tokens=4096, context=None, tools=None))]
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
        let model = model.to_string();

        let mut full_prompt = String::new();

        if let Some(ctx) = context {
            full_prompt.push_str("DOCUMENT CONTEXT:\n");
            full_prompt.push_str(&ctx);
            full_prompt.push_str("\n\n");
        }

        if !messages.is_empty() {
            full_prompt.push_str("CONVERSATION HISTORY:\n");
            for (role, content) in &messages[..messages.len().saturating_sub(1)] {
                full_prompt.push_str(&format!("{}: {}\n\n", role, content));
            }

            full_prompt.push_str("CURRENT QUESTION:\n");
            if let Some((_, content)) = messages.last() {
                full_prompt.push_str(content);
            }
        }

        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;

            let options = serde_json::json!({
                "temperature": temperature,
                "max_tokens": max_tokens,
            });

            engine
                .ai_complete(&model, &full_prompt, Some(options), tools)
                .await
        });

        result
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
            .map(|s| s.replace("\\_", "_"))
    }

    #[pyo3(signature = (messages, model="llama3.1-70b", temperature=0.7, max_tokens=4096, context=None, tools=None))]
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
        let model = model.to_string();
        let runtime = Arc::clone(&self.runtime);

        let mut rust_messages: Vec<Message> = messages
            .into_iter()
            .map(|(role, content)| {
                if role == "user" || role == "human" {
                    Message::new_user(content)
                } else if role == "assistant" || role == "ai" {
                    Message::new_assistant(content)
                } else if role == "system" {
                    Message::new_system(content)
                } else {
                    // Default to user, or handle other roles as needed.
                    // For tool calls, the python code sends a 'system' message.
                    Message::new_user(content)
                }
            })
            .collect();

        if let Some(ctx) = context {
            let system_message = Message::new_system(format!("DOCUMENT CONTEXT:\n{}", ctx));
            rust_messages.insert(0, system_message);
        }

        let result = self.runtime.block_on(async move {
            let mut engine = engine.lock().await;
            let mut agent_client = engine.create_agents_client()?;
            agent_client
                .stream_complete(
                    &model,
                    rust_messages,
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

    /// Close the cache connection
    fn close(&self) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        self.runtime
            .block_on(async move {
                let mut engine = engine.lock().await;
                engine.close().await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    /// Create streaming agent client
    fn create_agent(&self) -> PyResult<SnowflakeAgent> {
        let engine = Arc::clone(&self.engine);
        let runtime = Arc::clone(&self.runtime);

        let agent_client = self
            .runtime
            .block_on(async move {
                let mut engine = engine.lock().await;
                engine.create_agents_client()
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        Ok(SnowflakeAgent {
            client: Arc::new(Mutex::new(agent_client)),
            runtime,
        })
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
            engine.get_current_database().to_string()
        });
        Ok(result)
    }

    /// Get current schema
    fn get_current_schema(&self) -> PyResult<String> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.get_current_schema().to_string()
        });
        Ok(result)
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

    /// List all stages accessible to the user
    fn list_stages(&self) -> PyResult<Vec<String>> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let engine = engine.lock().await;
            engine.list_stages().await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// List files in a specific stage
    fn list_files_in_stage(&self, stage_name: String) -> PyResult<Vec<String>> {
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

            engine.list_files_in_stage(stage).await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Set conversation-specific data in the cache
    #[pyo3(signature = (conversation_id, key, value))]
    fn set_conversation_data(
        &self,
        conversation_id: String,
        key: String,
        value: String,
    ) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        self.runtime
            .block_on(async move {
                let cache = {
                    let engine = engine.lock().await;
                    engine.cache.clone()
                };
                let guard = cache.lock().await;
                guard
                    .set_conversation_data(conversation_id, key, value)
                    .await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    /// Get conversation-specific data from the cache
    #[pyo3(signature = (conversation_id, key))]
    fn get_conversation_data(
        &self,
        conversation_id: String,
        key: String,
    ) -> PyResult<Option<String>> {
        let engine = Arc::clone(&self.engine);
        let result = self
            .runtime
            .block_on(async move {
                let cache = {
                    let engine = engine.lock().await;
                    engine.cache.clone()
                };
                let guard = cache.lock().await;
                guard.get_conversation_data(&conversation_id, &key).await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result)
    }

    /// Delete conversation-specific data from the cache
    #[pyo3(signature = (conversation_id, key))]
    fn delete_conversation_data(&self, conversation_id: String, key: String) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        self.runtime
            .block_on(async move {
                let cache = {
                    let engine = engine.lock().await;
                    engine.cache.clone()
                };
                let guard = cache.lock().await;
                guard.delete_conversation_data(&conversation_id, &key).await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    /// List all conversation-specific data for a conversation
    #[pyo3(signature = (conversation_id))]
    fn list_conversation_data(&self, conversation_id: String) -> PyResult<Vec<(String, String)>> {
        let engine = Arc::clone(&self.engine);
        let result = self.runtime.block_on(async move {
            let cache = {
                let engine = engine.lock().await;
                engine.cache.clone()
            };
            let guard = cache.lock().await;
            guard.list_conversation_data(&conversation_id).await
        });
        result.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Clear conversation (delete all messages)
    #[pyo3(signature = (conversation_id))]
    fn clear_conversation(&self, conversation_id: String) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        self.runtime
            .block_on(async move {
                let cache = {
                    let engine = engine.lock().await;
                    engine.cache.clone()
                };
                let guard = cache.lock().await;
                guard.clear_conversation(&conversation_id).await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    /// Add a message to the conversation cache
    #[pyo3(signature = (conversation_id, message_id, role, content))]
    fn add_message(
        &self,
        conversation_id: String,
        message_id: String,
        role: String,
        content: String,
    ) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        self.runtime
            .block_on(async move {
                let cache = {
                    let engine = engine.lock().await;
                    engine.cache.clone()
                };
                let guard = cache.lock().await;
                guard
                    .add_message(conversation_id, message_id, role, content)
                    .await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    /// Get all messages for a conversation (ordered by timestamp)
    fn get_messages(&self, conversation_id: String) -> PyResult<Vec<(String, String, String)>> {
        let engine = Arc::clone(&self.engine);
        self.runtime
            .block_on(async move {
                let cache = {
                    let engine = engine.lock().await;
                    engine.cache.clone()
                };
                let guard = cache.lock().await;
                guard.get_messages(&conversation_id).await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Get conversation history from the cache (DEPRECATED - use get_messages instead)
    #[pyo3(signature = (conversation_id))]
    fn get_conversation_history(&self, conversation_id: String) -> PyResult<Option<String>> {
        let engine = Arc::clone(&self.engine);
        let result = self
            .runtime
            .block_on(async move {
                let cache = {
                    let engine = engine.lock().await;
                    engine.cache.clone()
                };
                let guard = cache.lock().await;
                guard
                    .get_conversation_data(&conversation_id, "conversation_history")
                    .await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(result)
    }

    /// Set conversation history in the cache (DEPRECATED - use add_message instead)
    #[pyo3(signature = (conversation_id, history))]
    fn set_conversation_history(&self, conversation_id: String, history: String) -> PyResult<()> {
        let engine = Arc::clone(&self.engine);
        self.runtime
            .block_on(async move {
                let cache = {
                    let engine = engine.lock().await;
                    engine.cache.clone()
                };
                let guard = cache.lock().await;
                guard
                    .set_conversation_data(
                        conversation_id,
                        "conversation_history".to_string(),
                        history,
                    )
                    .await
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(())
    }

    // Tool definitions
    #[staticmethod]
    fn get_table_sample_data_tool() -> agents::Tool {
        agents::Tool {
            tool_spec: agents::ToolSpec {
                tool_type: "generic".to_string(),
                name: "get_table_sample_data".to_string(),
                description: "Retrieves sample data from a specified table. Use this to understand the content and structure of a table.".to_string(),
                input_schema: agents::ToolInputSchema {
                    schema_type: "object".to_string(),
                    description: None,
                    properties: Some(
                        vec![
                            (
                                "table_name".to_string(),
                                agents::ToolInputSchema {
                                    schema_type: "string".to_string(),
                                    description: Some("The fully qualified name of the table (e.g., DATABASE.SCHEMA.TABLE_NAME).".to_string()),
                                    properties: None,
                                    items: None,
                                    required: None,
                                },
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                    items: None,
                    required: Some(vec!["table_name".to_string()]),
                },
            },
        }
    }

    #[staticmethod]
    fn get_table_definition_tool() -> agents::Tool {
        agents::Tool {
            tool_spec: agents::ToolSpec {
                tool_type: "generic".to_string(),
                name: "get_table_definition".to_string(),
                description: "Retrieves the column definitions (schema) for a specified table. Use this to understand the structure and data types of a table.".to_string(),
                input_schema: agents::ToolInputSchema {
                    schema_type: "object".to_string(),
                    description: None,
                    properties: Some(
                        vec![(
                            "table_name".to_string(),
                            agents::ToolInputSchema {
                                schema_type: "string".to_string(),
                                description: Some("The fully qualified name of the table (e.g., DATABASE.SCHEMA.TABLE_NAME).".to_string()),
                                properties: None,
                                items: None,
                                required: None,
                            },
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    items: None,
                    required: Some(vec!["table_name".to_string()]),
                },
            },
        }
    }

    #[staticmethod]
    fn get_multiple_table_definitions_tool() -> agents::Tool {
        agents::Tool {
            tool_spec: agents::ToolSpec {
                tool_type: "generic".to_string(),
                name: "get_multiple_table_definitions".to_string(),
                description:
                    "Retrieves the column definitions (schema) for a list of specified tables."
                        .to_string(),
                input_schema: agents::ToolInputSchema {
                    schema_type: "object".to_string(),
                    description: None,
                    properties: Some(
                        vec![(
                            "table_names".to_string(),
                            agents::ToolInputSchema {
                                schema_type: "array".to_string(),
                                description: Some(
                                    "The list of fully qualified table names.".to_string(),
                                ),
                                properties: None,
                                items: Some(Box::new(agents::ToolInputSchema {
                                    schema_type: "string".to_string(),
                                    description: None,
                                    properties: None,
                                    items: None,
                                    required: None,
                                })),
                                required: None,
                            },
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    items: None,
                    required: Some(vec!["table_names".to_string()]),
                },
            },
        }
    }

    #[staticmethod]
    fn list_databases_tool() -> agents::Tool {
        agents::Tool {
            tool_spec: agents::ToolSpec {
                tool_type: "generic".to_string(),
                name: "list_databases".to_string(),
                description: "Lists all databases accessible to the user.".to_string(),
                input_schema: agents::ToolInputSchema {
                    schema_type: "object".to_string(),
                    description: None,
                    properties: Some(std::collections::HashMap::new()), // No properties
                    items: None,
                    required: None,
                },
            },
        }
    }

    #[staticmethod]
    fn list_schemas_tool() -> agents::Tool {
        agents::Tool {
            tool_spec: agents::ToolSpec {
                tool_type: "generic".to_string(),
                name: "list_schemas".to_string(),
                description: "Lists all schemas in a specified database. If no database is specified, lists schemas in the current database.".to_string(),
                input_schema: agents::ToolInputSchema {
                    schema_type: "object".to_string(),
                    description: None,
                    properties: Some(
                        vec![
                            (
                                "database".to_string(),
                                agents::ToolInputSchema {
                                    schema_type: "string".to_string(),
                                    description: Some("The name of the database to list schemas for.".to_string()),
                                    properties: None,
                                    items: None,
                                    required: None, // Optional
                                },
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                    items: None,
                    required: None,
                },
            },
        }
    }

    #[staticmethod]
    fn list_tables_in_tool() -> agents::Tool {
        agents::Tool {
            tool_spec: agents::ToolSpec {
                tool_type: "generic".to_string(),
                name: "list_tables_in".to_string(),
                description: "Lists all tables in a specified database and schema.".to_string(),
                input_schema: agents::ToolInputSchema {
                    schema_type: "object".to_string(),
                    description: None,
                    properties: Some(
                        vec![
                            (
                                "database".to_string(),
                                agents::ToolInputSchema {
                                    schema_type: "string".to_string(),
                                    description: Some("The name of the database.".to_string()),
                                    properties: None,
                                    items: None,
                                    required: None,
                                },
                            ),
                            (
                                "schema".to_string(),
                                agents::ToolInputSchema {
                                    schema_type: "string".to_string(),
                                    description: Some(
                                        "The name of the schema within the database.".to_string(),
                                    ),
                                    properties: None,
                                    items: None,
                                    required: None,
                                },
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                    items: None,
                    required: Some(vec!["database".to_string(), "schema".to_string()]),
                },
            },
        }
    }

    #[staticmethod]
    fn validate_query_tool() -> agents::Tool {
        agents::Tool {
            tool_spec: agents::ToolSpec {
                tool_type: "generic".to_string(),
                name: "validate_query".to_string(),
                description: "Validates a given SQL query without executing it. Returns an error if the query is invalid.".to_string(),
                input_schema: agents::ToolInputSchema {
                    schema_type: "object".to_string(),
                    description: None,
                    properties: Some(
                        vec![
                            (
                                "query".to_string(),
                                agents::ToolInputSchema {
                                    schema_type: "string".to_string(),
                                    description: Some("The SQL query string to validate.".to_string()),
                                    properties: None,
                                    items: None,
                                    required: None,
                                },
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                    items: None,
                    required: Some(vec!["query".to_string()]),
                },
            },
        }
    }

    #[staticmethod]
    fn execute_query_tool() -> agents::Tool {
        agents::Tool {
            tool_spec: agents::ToolSpec {
                tool_type: "generic".to_string(),
                name: "execute_query".to_string(),
                description: "Executes a given SQL query and returns the results. Use this to retrieve data from the database.".to_string(),
                input_schema: agents::ToolInputSchema {
                    schema_type: "object".to_string(),
                    description: None,
                    properties: Some(
                        vec![
                            (
                                "query".to_string(),
                                agents::ToolInputSchema {
                                    schema_type: "string".to_string(),
                                    description: Some("The SQL query string to execute.".to_string()),
                                    properties: None,
                                    items: None,
                                    required: None,
                                },
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                    items: None,
                    required: Some(vec!["query".to_string()]),
                },
            },
        }
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
    #[pyo3(signature = (message, model="llama3.1-70b", temperature=None, max_tokens=None))]
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

    #[pyo3(signature = (message, model="llama3.1-70b", temperature=None, max_tokens=None, use_history=true))]
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

    #[pyo3(signature = (message, model="llama3.1-70b", temperature=None, max_tokens=None, use_history=true, tools=None, tool_choice=None))]
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

    #[pyo3(signature = (message, model="llama3.1-70b", temperature=None, max_tokens=None, use_history=true))]
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
    m.add_class::<agents::CachedMessage>()?;
    Ok(())
}
