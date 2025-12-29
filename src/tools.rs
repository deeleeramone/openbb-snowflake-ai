//! Tool registry and execution framework for the CLI.
//!
//! Provides a registry of all available tools and a dispatcher to execute them.
//! Tools are defined with JSON schemas matching the Snowflake Cortex API format.

use crate::engine::SnowflakeEngine;
use crate::events::EventEmitter;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

// ============================================================================
// Tool Definition Types
// ============================================================================

/// Tool category for organization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCategory {
    Query,
    Database,
    AiFunction,
    CortexNlp,
    Document,
    Chart,
    Pagination,
}

/// Tool definition with metadata
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub category: ToolCategory,
    pub schema: Value,
}

/// Result of tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    pub success: bool,
    pub result: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolExecutionResult {
    pub fn success(result: Value) -> Self {
        Self {
            success: true,
            result,
            error: None,
        }
    }

    pub fn failure(error: String) -> Self {
        Self {
            success: false,
            result: Value::Null,
            error: Some(error),
        }
    }
}

// ============================================================================
// Tool Registry
// ============================================================================

/// Registry of all available tools
pub struct ToolRegistry {
    tools: HashMap<String, ToolDefinition>,
}

impl ToolRegistry {
    /// Create a new tool registry with all tools registered
    pub fn new() -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
        };
        registry.register_all_tools();
        registry
    }

    /// Register all available tools
    fn register_all_tools(&mut self) {
        // Query tools
        self.register_query_tools();
        // Database tools
        self.register_database_tools();
        // AI function tools
        self.register_ai_function_tools();
        // Cortex NLP tools
        self.register_cortex_nlp_tools();
        // Document tools
        self.register_document_tools();
        // Chart tools
        self.register_chart_tools();
        // Pagination tools
        self.register_pagination_tools();
    }

    fn register_query_tools(&mut self) {
        self.tools.insert(
            "text2sql".to_string(),
            ToolDefinition {
                name: "text2sql".to_string(),
                description: "Generate SQL from natural language description. Returns the generated SQL query without executing it.".to_string(),
                category: ToolCategory::Query,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Natural language description of the data you want to query"
                        }
                    },
                    "required": ["prompt"]
                }),
            },
        );

        self.tools.insert(
            "execute_query".to_string(),
            ToolDefinition {
                name: "execute_query".to_string(),
                description:
                    "Execute a SQL query and return the results as JSON. Use for SELECT statements."
                        .to_string(),
                category: ToolCategory::Query,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The SQL query to execute"
                        }
                    },
                    "required": ["query"]
                }),
            },
        );

        self.tools.insert(
            "execute_statement".to_string(),
            ToolDefinition {
                name: "execute_statement".to_string(),
                description: "Execute a SQL statement (DDL/DML) without returning results. Use for CREATE, INSERT, UPDATE, DELETE, etc.".to_string(),
                category: ToolCategory::Query,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": "The SQL statement to execute"
                        }
                    },
                    "required": ["statement"]
                }),
            },
        );

        self.tools.insert(
            "validate_query".to_string(),
            ToolDefinition {
                name: "validate_query".to_string(),
                description: "Validate a SQL query for syntax errors without executing it."
                    .to_string(),
                category: ToolCategory::Query,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The SQL query to validate"
                        }
                    },
                    "required": ["query"]
                }),
            },
        );
    }

    fn register_database_tools(&mut self) {
        self.tools.insert(
            "list_databases".to_string(),
            ToolDefinition {
                name: "list_databases".to_string(),
                description: "List all accessible databases.".to_string(),
                category: ToolCategory::Database,
                schema: json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        );

        self.tools.insert(
            "list_schemas".to_string(),
            ToolDefinition {
                name: "list_schemas".to_string(),
                description: "List all schemas in a database.".to_string(),
                category: ToolCategory::Database,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "Database name (optional, defaults to current)"
                        }
                    }
                }),
            },
        );

        self.tools.insert(
            "list_tables_in".to_string(),
            ToolDefinition {
                name: "list_tables_in".to_string(),
                description: "List all tables in a specific database and schema.".to_string(),
                category: ToolCategory::Database,
                schema: json!({
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
                }),
            },
        );

        self.tools.insert(
            "get_table_schema".to_string(),
            ToolDefinition {
                name: "get_table_schema".to_string(),
                description: "Get the column definitions for a table.".to_string(),
                category: ToolCategory::Database,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Fully qualified table name (database.schema.table) or just table name"
                        }
                    },
                    "required": ["table_name"]
                }),
            },
        );

        self.tools.insert(
            "get_table_sample_data".to_string(),
            ToolDefinition {
                name: "get_table_sample_data".to_string(),
                description: "Get sample rows from a table to understand its data.".to_string(),
                category: ToolCategory::Database,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Fully qualified table name or just table name"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of sample rows (default: 5)"
                        }
                    },
                    "required": ["table_name"]
                }),
            },
        );

        self.tools.insert(
            "get_multiple_table_definitions".to_string(),
            ToolDefinition {
                name: "get_multiple_table_definitions".to_string(),
                description: "Get schema definitions for multiple tables at once.".to_string(),
                category: ToolCategory::Database,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "table_names": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "List of fully qualified table names"
                        }
                    },
                    "required": ["table_names"]
                }),
            },
        );

        self.tools.insert(
            "list_semantic_views".to_string(),
            ToolDefinition {
                name: "list_semantic_views".to_string(),
                description: "List semantic views available for text2sql operations.".to_string(),
                category: ToolCategory::Database,
                schema: json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        );

        self.tools.insert(
            "describe_table".to_string(),
            ToolDefinition {
                name: "describe_table".to_string(),
                description:
                    "Get detailed description of a table including columns, types, and comments."
                        .to_string(),
                category: ToolCategory::Database,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Table name to describe"
                        }
                    },
                    "required": ["table_name"]
                }),
            },
        );
    }

    fn register_ai_function_tools(&mut self) {
        self.tools.insert(
            "ai_filter".to_string(),
            ToolDefinition {
                name: "ai_filter".to_string(),
                description: "AI-powered boolean filtering on text or images. Returns true/false for each row based on a natural language condition.".to_string(),
                category: ToolCategory::AiFunction,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "condition": {
                            "type": "string",
                            "description": "Natural language condition to evaluate"
                        },
                        "column": {
                            "type": "string",
                            "description": "Column name containing text or image data"
                        },
                        "table": {
                            "type": "string",
                            "description": "Table name to filter"
                        }
                    },
                    "required": ["condition", "column", "table"]
                }),
            },
        );

        self.tools.insert(
            "ai_agg".to_string(),
            ToolDefinition {
                name: "ai_agg".to_string(),
                description: "AI-powered aggregation over text data. Summarizes or extracts insights from multiple rows.".to_string(),
                category: ToolCategory::AiFunction,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "Natural language instruction for aggregation"
                        },
                        "column": {
                            "type": "string",
                            "description": "Column name to aggregate"
                        },
                        "table": {
                            "type": "string",
                            "description": "Table name"
                        },
                        "group_by": {
                            "type": "string",
                            "description": "Optional column to group by"
                        }
                    },
                    "required": ["instruction", "column", "table"]
                }),
            },
        );

        self.tools.insert(
            "ai_summarize_agg".to_string(),
            ToolDefinition {
                name: "ai_summarize_agg".to_string(),
                description: "AI-powered aggregation with summarization. Combines rows and produces a summary.".to_string(),
                category: ToolCategory::AiFunction,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "Natural language instruction for summarization"
                        },
                        "column": {
                            "type": "string",
                            "description": "Column name to summarize"
                        },
                        "table": {
                            "type": "string",
                            "description": "Table name"
                        }
                    },
                    "required": ["instruction", "column", "table"]
                }),
            },
        );

        self.tools.insert(
            "extract_answer".to_string(),
            ToolDefinition {
                name: "extract_answer".to_string(),
                description: "Extract specific facts or answers from documents using AI."
                    .to_string(),
                category: ToolCategory::AiFunction,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Question to answer from the document"
                        },
                        "document": {
                            "type": "string",
                            "description": "Document name or content to extract from"
                        }
                    },
                    "required": ["question", "document"]
                }),
            },
        );
    }

    fn register_cortex_nlp_tools(&mut self) {
        self.tools.insert(
            "sentiment".to_string(),
            ToolDefinition {
                name: "sentiment".to_string(),
                description:
                    "Analyze sentiment of text. Returns a score from -1 (negative) to 1 (positive)."
                        .to_string(),
                category: ToolCategory::CortexNlp,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to analyze"
                        }
                    },
                    "required": ["text"]
                }),
            },
        );

        self.tools.insert(
            "summarize".to_string(),
            ToolDefinition {
                name: "summarize".to_string(),
                description: "Summarize text content.".to_string(),
                category: ToolCategory::CortexNlp,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to summarize"
                        }
                    },
                    "required": ["text"]
                }),
            },
        );

        self.tools.insert(
            "translate".to_string(),
            ToolDefinition {
                name: "translate".to_string(),
                description: "Translate text to another language.".to_string(),
                category: ToolCategory::CortexNlp,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to translate"
                        },
                        "target_language": {
                            "type": "string",
                            "description": "Target language code (e.g., 'es', 'fr', 'de')"
                        }
                    },
                    "required": ["text", "target_language"]
                }),
            },
        );
    }

    fn register_document_tools(&mut self) {
        self.tools.insert(
            "read_document".to_string(),
            ToolDefinition {
                name: "read_document".to_string(),
                description: "Read specific pages from a parsed document.".to_string(),
                category: ToolCategory::Document,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "document_name": {
                            "type": "string",
                            "description": "Name of the document to read"
                        },
                        "pages": {
                            "type": "array",
                            "items": { "type": "integer" },
                            "description": "Page numbers to read (1-indexed)"
                        }
                    },
                    "required": ["document_name"]
                }),
            },
        );

        self.tools.insert(
            "search_document".to_string(),
            ToolDefinition {
                name: "search_document".to_string(),
                description: "Semantic search across documents using vector embeddings."
                    .to_string(),
                category: ToolCategory::Document,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "document_name": {
                            "type": "string",
                            "description": "Optional: limit search to specific document"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 5)"
                        }
                    },
                    "required": ["query"]
                }),
            },
        );

        self.tools.insert(
            "get_document_images".to_string(),
            ToolDefinition {
                name: "get_document_images".to_string(),
                description: "Extract images from document pages.".to_string(),
                category: ToolCategory::Document,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "document_name": {
                            "type": "string",
                            "description": "Name of the document"
                        },
                        "page": {
                            "type": "integer",
                            "description": "Page number to extract images from"
                        }
                    },
                    "required": ["document_name", "page"]
                }),
            },
        );

        self.tools.insert(
            "ocr_image".to_string(),
            ToolDefinition {
                name: "ocr_image".to_string(),
                description: "Extract text from images using OCR and vision models.".to_string(),
                category: ToolCategory::Document,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Stage path to the image file"
                        }
                    },
                    "required": ["image_path"]
                }),
            },
        );
    }

    fn register_chart_tools(&mut self) {
        self.tools.insert(
            "render_chart".to_string(),
            ToolDefinition {
                name: "render_chart".to_string(),
                description:
                    "Generate an interactive chart from data. Outputs an HTML file with Chart.js."
                        .to_string(),
                category: ToolCategory::Chart,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "pie", "donut", "scatter", "area"],
                            "description": "Type of chart to render"
                        },
                        "data": {
                            "type": "object",
                            "description": "Chart data with labels and datasets"
                        },
                        "title": {
                            "type": "string",
                            "description": "Chart title"
                        },
                        "x_label": {
                            "type": "string",
                            "description": "X-axis label"
                        },
                        "y_label": {
                            "type": "string",
                            "description": "Y-axis label"
                        }
                    },
                    "required": ["chart_type", "data"]
                }),
            },
        );
    }

    fn register_pagination_tools(&mut self) {
        self.tools.insert(
            "continue_output".to_string(),
            ToolDefinition {
                name: "continue_output".to_string(),
                description: "Continue a truncated output from a previous tool call.".to_string(),
                category: ToolCategory::Pagination,
                schema: json!({
                    "type": "object",
                    "properties": {
                        "continuation_token": {
                            "type": "string",
                            "description": "Token from previous truncated output"
                        }
                    },
                    "required": ["continuation_token"]
                }),
            },
        );
    }

    /// Get all tool definitions as JSON array for Cortex API
    pub fn get_tools_json(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|tool| {
                json!({
                    "tool_spec": {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.schema
                    }
                })
            })
            .collect()
    }

    /// Get a specific tool definition
    pub fn get_tool(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    /// Get all tool names
    pub fn list_tools(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Get tools by category
    pub fn get_tools_by_category(&self, category: ToolCategory) -> Vec<&ToolDefinition> {
        self.tools
            .values()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Format tool definitions as a human-readable string for system prompt
    pub fn format_tool_definitions(&self) -> String {
        let mut output = String::new();

        // Group tools by category
        let categories = [
            (ToolCategory::Query, "Query Tools"),
            (ToolCategory::Database, "Database Tools"),
            (ToolCategory::AiFunction, "AI Function Tools"),
            (ToolCategory::CortexNlp, "Cortex NLP Tools"),
            (ToolCategory::Document, "Document Tools"),
            (ToolCategory::Chart, "Chart Tools"),
            (ToolCategory::Pagination, "Pagination Tools"),
        ];

        for (category, label) in categories {
            let tools = self.get_tools_by_category(category);
            if !tools.is_empty() {
                output.push_str(&format!("\n#### {}\n", label));
                for tool in tools {
                    output.push_str(&format!("- **{}**: {}\n", tool.name, tool.description));
                    // Include parameters if they exist
                    if let Some(props) = tool.schema.get("properties") {
                        if let Some(obj) = props.as_object() {
                            if !obj.is_empty() {
                                output.push_str("  Parameters: ");
                                let params: Vec<String> =
                                    obj.keys().map(|k| format!("`{}`", k)).collect();
                                output.push_str(&params.join(", "));
                                output.push('\n');
                            }
                        }
                    }
                }
            }
        }

        output
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tool Executor
// ============================================================================

/// Executor for running tools
pub struct ToolExecutor {
    registry: ToolRegistry,
}

impl ToolExecutor {
    pub fn new() -> Self {
        Self {
            registry: ToolRegistry::new(),
        }
    }

    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Qualify a table name with current database/schema context if not already qualified.
    /// A fully qualified name has format: DATABASE.SCHEMA.TABLE
    async fn qualify_table_name(
        engine_guard: &tokio::sync::MutexGuard<'_, SnowflakeEngine>,
        table_name: &str,
    ) -> String {
        let parts: Vec<&str> = table_name.split('.').collect();

        match parts.len() {
            // Already fully qualified (DATABASE.SCHEMA.TABLE)
            3 => table_name.to_string(),
            // Partially qualified (SCHEMA.TABLE) - add database
            2 => {
                let current_db = engine_guard.get_database().await.unwrap_or_default();
                if current_db.is_empty() {
                    table_name.to_string()
                } else {
                    format!("{}.{}", current_db, table_name)
                }
            }
            // Just table name - add database and schema
            1 => {
                let current_db = engine_guard.get_database().await.unwrap_or_default();
                let current_schema = engine_guard.get_schema().await.unwrap_or_default();
                if current_db.is_empty() || current_schema.is_empty() {
                    table_name.to_string()
                } else {
                    format!("{}.{}.{}", current_db, current_schema, table_name)
                }
            }
            _ => table_name.to_string(),
        }
    }

    /// Execute a tool by name with given arguments
    pub async fn execute(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        tool_name: &str,
        arguments: Value,
        _emitter: &EventEmitter,
    ) -> ToolExecutionResult {
        // NOTE: tool_use and tool_result events are emitted by the agentic loop,
        // not here, to avoid duplicate emissions.

        match tool_name {
            // Query tools
            "text2sql" => self.execute_text2sql(engine, &arguments).await,
            "execute_query" => self.execute_query(engine, &arguments).await,
            "execute_statement" => self.execute_statement(engine, &arguments).await,
            "validate_query" => self.execute_validate_query(engine, &arguments).await,

            // Database tools
            "list_databases" => self.execute_list_databases(engine).await,
            "list_schemas" => self.execute_list_schemas(engine, &arguments).await,
            "list_tables_in" => self.execute_list_tables_in(engine, &arguments).await,
            "get_table_schema" => self.execute_get_table_schema(engine, &arguments).await,
            "get_table_sample_data" => self.execute_get_table_sample_data(engine, &arguments).await,
            "get_multiple_table_definitions" => {
                self.execute_get_multiple_table_definitions(engine, &arguments)
                    .await
            }
            "list_semantic_views" => self.execute_list_semantic_views(engine).await,
            "describe_table" => self.execute_describe_table(engine, &arguments).await,

            // AI function tools
            "ai_filter" => self.execute_ai_filter(engine, &arguments).await,
            "ai_agg" => self.execute_ai_agg(engine, &arguments).await,
            "ai_summarize_agg" => self.execute_ai_summarize_agg(engine, &arguments).await,
            "extract_answer" => self.execute_extract_answer(engine, &arguments).await,

            // Cortex NLP tools
            "sentiment" => self.execute_sentiment(engine, &arguments).await,
            "summarize" => self.execute_summarize(engine, &arguments).await,
            "translate" => self.execute_translate(engine, &arguments).await,

            // Document tools
            "read_document" => self.execute_read_document(engine, &arguments).await,
            "search_document" => self.execute_search_document(engine, &arguments).await,
            "get_document_images" => self.execute_get_document_images(engine, &arguments).await,
            "ocr_image" => self.execute_ocr_image(engine, &arguments).await,

            // Chart tools
            "render_chart" => self.execute_render_chart(&arguments).await,

            // Pagination tools
            "continue_output" => self.execute_continue_output(&arguments).await,

            _ => ToolExecutionResult::failure(format!("Unknown tool: {}", tool_name)),
        }
    }

    // ========================================================================
    // Query Tool Implementations
    // ========================================================================

    async fn execute_text2sql(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let prompt = match args.get("prompt").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: prompt".to_string(),
                )
            }
        };

        let mut engine_guard = engine.lock().await;
        match engine_guard
            .execute_with_cortex_context(prompt, false)
            .await
        {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_query(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => {
                return ToolExecutionResult::failure("Missing required argument: query".to_string())
            }
        };

        let engine_guard = engine.lock().await;
        match engine_guard.execute_query(query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_statement(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let statement = match args.get("statement").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: statement".to_string(),
                )
            }
        };

        let engine_guard = engine.lock().await;
        match engine_guard.execute_statement(statement).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_validate_query(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let query = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => {
                return ToolExecutionResult::failure("Missing required argument: query".to_string())
            }
        };

        let engine_guard = engine.lock().await;
        match engine_guard.validate_query(query).await {
            Ok(()) => ToolExecutionResult::success(json!({ "valid": true })),
            Err(e) => ToolExecutionResult::success(json!({ "valid": false, "error": e })),
        }
    }

    // ========================================================================
    // Database Tool Implementations
    // ========================================================================

    async fn execute_list_databases(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
    ) -> ToolExecutionResult {
        let engine_guard = engine.lock().await;
        match engine_guard.list_databases().await {
            Ok(databases) => ToolExecutionResult::success(json!({ "databases": databases })),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_list_schemas(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let engine_guard = engine.lock().await;

        // Get current database from engine as default
        let current_db = engine_guard.get_database().await.unwrap_or_default();

        // Use provided database OR fall back to current context
        let database = args
            .get("database")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .or_else(|| {
                if current_db.is_empty() {
                    None
                } else {
                    Some(current_db)
                }
            });

        match engine_guard.list_schemas(database.as_deref()).await {
            Ok(schemas) => ToolExecutionResult::success(json!({
                "database": database.unwrap_or_else(|| "current".to_string()),
                "schemas": schemas
            })),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_list_tables_in(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let engine_guard = engine.lock().await;

        // Get current context from engine
        let current_db = engine_guard.get_database().await.unwrap_or_default();
        let current_schema = engine_guard.get_schema().await.unwrap_or_default();

        // Use provided values OR fall back to current context
        let database = args
            .get("database")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(&current_db);
        let schema = args
            .get("schema")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .unwrap_or(&current_schema);

        if database.is_empty() || schema.is_empty() {
            return ToolExecutionResult::failure(
                "No database/schema context available. Use /use_database and /use_schema first."
                    .to_string(),
            );
        }

        match engine_guard.list_tables_in(database, schema).await {
            Ok(tables) => ToolExecutionResult::success(json!({
                "database": database,
                "schema": schema,
                "tables": tables
            })),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_get_table_schema(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let table_name = match args.get("table_name").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: table_name".to_string(),
                )
            }
        };

        let engine_guard = engine.lock().await;

        // Qualify the table name with current context if not already qualified
        let qualified_name = Self::qualify_table_name(&engine_guard, table_name).await;

        match engine_guard.get_table_info(&qualified_name).await {
            Ok(info) => ToolExecutionResult::success(json!({
                "table": qualified_name,
                "table_info": info
            })),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_get_table_sample_data(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let table_name = match args.get("table_name").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: table_name".to_string(),
                )
            }
        };

        let engine_guard = engine.lock().await;

        // Qualify the table name with current context
        let qualified_name = Self::qualify_table_name(&engine_guard, table_name).await;

        let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(5);
        let query = format!("SELECT * FROM {} LIMIT {}", qualified_name, limit);

        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(json!({
                "table": qualified_name,
                "data": result
            })),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_get_multiple_table_definitions(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let table_names = match args.get("table_names").and_then(|v| v.as_array()) {
            Some(arr) => arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>(),
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: table_names".to_string(),
                )
            }
        };

        let engine_guard = engine.lock().await;
        let mut results = Vec::new();

        for table_name in table_names {
            // Qualify each table name with current context
            let qualified_name = Self::qualify_table_name(&engine_guard, table_name).await;
            match engine_guard.get_table_info(&qualified_name).await {
                Ok(info) => results.push(json!({ "table": qualified_name, "schema": info })),
                Err(e) => results.push(json!({ "table": qualified_name, "error": e })),
            }
        }

        ToolExecutionResult::success(json!({ "tables": results }))
    }

    async fn execute_list_semantic_views(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
    ) -> ToolExecutionResult {
        let mut engine_guard = engine.lock().await;
        let views = engine_guard.get_semantic_views().await;
        // Serialize the views to JSON
        match serde_json::to_value(&views) {
            Ok(json_views) => ToolExecutionResult::success(json!({ "semantic_views": json_views })),
            Err(e) => ToolExecutionResult::failure(format!("Failed to serialize views: {}", e)),
        }
    }

    async fn execute_describe_table(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let table_name = match args.get("table_name").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: table_name".to_string(),
                )
            }
        };

        let engine_guard = engine.lock().await;

        // Qualify the table name with current context
        let qualified_name = Self::qualify_table_name(&engine_guard, table_name).await;

        let query = format!("DESCRIBE TABLE {}", qualified_name);
        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(json!({
                "table": qualified_name,
                "description": result
            })),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    // ========================================================================
    // AI Function Tool Implementations
    // ========================================================================

    async fn execute_ai_filter(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let condition = match args.get("condition").and_then(|v| v.as_str()) {
            Some(c) => c.replace('\'', "''"),
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: condition".to_string(),
                )
            }
        };
        let column = match args.get("column").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: column".to_string(),
                )
            }
        };
        let table = match args.get("table").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return ToolExecutionResult::failure("Missing required argument: table".to_string())
            }
        };

        let query = format!(
            "SELECT *, AI_FILTER('{}', {}) AS ai_filter_result FROM {} WHERE AI_FILTER('{}', {})",
            condition, column, table, condition, column
        );

        let engine_guard = engine.lock().await;
        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_ai_agg(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let instruction = match args.get("instruction").and_then(|v| v.as_str()) {
            Some(i) => i.replace('\'', "''"),
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: instruction".to_string(),
                )
            }
        };
        let column = match args.get("column").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: column".to_string(),
                )
            }
        };
        let table = match args.get("table").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return ToolExecutionResult::failure("Missing required argument: table".to_string())
            }
        };
        let group_by = args.get("group_by").and_then(|v| v.as_str());

        let query = if let Some(gb) = group_by {
            format!(
                "SELECT {}, AI_AGG('{}', {}) AS ai_agg_result FROM {} GROUP BY {}",
                gb, instruction, column, table, gb
            )
        } else {
            format!(
                "SELECT AI_AGG('{}', {}) AS ai_agg_result FROM {}",
                instruction, column, table
            )
        };

        let engine_guard = engine.lock().await;
        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_ai_summarize_agg(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let instruction = match args.get("instruction").and_then(|v| v.as_str()) {
            Some(i) => i.replace('\'', "''"),
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: instruction".to_string(),
                )
            }
        };
        let column = match args.get("column").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: column".to_string(),
                )
            }
        };
        let table = match args.get("table").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return ToolExecutionResult::failure("Missing required argument: table".to_string())
            }
        };

        let query = format!(
            "SELECT AI_SUMMARIZE_AGG('{}', {}) AS summary FROM {}",
            instruction, column, table
        );

        let engine_guard = engine.lock().await;
        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_extract_answer(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let question = match args.get("question").and_then(|v| v.as_str()) {
            Some(q) => q.replace('\'', "''"),
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: question".to_string(),
                )
            }
        };
        let document = match args.get("document").and_then(|v| v.as_str()) {
            Some(d) => d.replace('\'', "''"),
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: document".to_string(),
                )
            }
        };

        let query = format!(
            "SELECT AI_EXTRACT('{}', '{}') AS answer",
            question, document
        );

        let engine_guard = engine.lock().await;
        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    // ========================================================================
    // Cortex NLP Tool Implementations
    // ========================================================================

    async fn execute_sentiment(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let text = match args.get("text").and_then(|v| v.as_str()) {
            Some(t) => t.replace('\'', "''"),
            None => {
                return ToolExecutionResult::failure("Missing required argument: text".to_string())
            }
        };

        let query = format!("SELECT SNOWFLAKE.CORTEX.SENTIMENT('{}') AS sentiment", text);

        let engine_guard = engine.lock().await;
        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_summarize(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let text = match args.get("text").and_then(|v| v.as_str()) {
            Some(t) => t.replace('\'', "''"),
            None => {
                return ToolExecutionResult::failure("Missing required argument: text".to_string())
            }
        };

        let query = format!("SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{}') AS summary", text);

        let engine_guard = engine.lock().await;
        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_translate(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let text = match args.get("text").and_then(|v| v.as_str()) {
            Some(t) => t.replace('\'', "''"),
            None => {
                return ToolExecutionResult::failure("Missing required argument: text".to_string())
            }
        };
        let target_language = match args.get("target_language").and_then(|v| v.as_str()) {
            Some(l) => l,
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: target_language".to_string(),
                )
            }
        };

        let query = format!(
            "SELECT SNOWFLAKE.CORTEX.TRANSLATE('{}', '', '{}') AS translation",
            text, target_language
        );

        let engine_guard = engine.lock().await;
        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    // ========================================================================
    // Document Tool Implementations
    // ========================================================================

    async fn execute_read_document(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let document_name = match args.get("document_name").and_then(|v| v.as_str()) {
            Some(d) => d.replace('\'', "''"),
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: document_name".to_string(),
                )
            }
        };

        let pages = args.get("pages").and_then(|v| v.as_array());

        let engine_guard = engine.lock().await;
        let user = match engine_guard.get_user().await {
            Ok(u) => u,
            Err(e) => return ToolExecutionResult::failure(e),
        };

        let schema = format!("OPENBB_AGENTS.USER_{}", user.to_uppercase());

        let query = if let Some(page_list) = pages {
            let page_nums: Vec<String> = page_list
                .iter()
                .filter_map(|v| v.as_i64())
                .map(|n| n.to_string())
                .collect();
            if page_nums.is_empty() {
                format!(
                    "SELECT PAGE_NUMBER, PAGE_CONTENT FROM {}.DOCUMENT_PARSE_RESULTS WHERE FILE_NAME = '{}' ORDER BY PAGE_NUMBER",
                    schema, document_name
                )
            } else {
                format!(
                    "SELECT PAGE_NUMBER, PAGE_CONTENT FROM {}.DOCUMENT_PARSE_RESULTS WHERE FILE_NAME = '{}' AND PAGE_NUMBER IN ({}) ORDER BY PAGE_NUMBER",
                    schema, document_name, page_nums.join(",")
                )
            }
        } else {
            format!(
                "SELECT PAGE_NUMBER, PAGE_CONTENT FROM {}.DOCUMENT_PARSE_RESULTS WHERE FILE_NAME = '{}' ORDER BY PAGE_NUMBER",
                schema, document_name
            )
        };

        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_search_document(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let query_text = match args.get("query").and_then(|v| v.as_str()) {
            Some(q) => q.replace('\'', "''"),
            None => {
                return ToolExecutionResult::failure("Missing required argument: query".to_string())
            }
        };

        let document_name = args.get("document_name").and_then(|v| v.as_str());
        let limit = args.get("limit").and_then(|v| v.as_i64()).unwrap_or(5);

        let engine_guard = engine.lock().await;
        let user = match engine_guard.get_user().await {
            Ok(u) => u,
            Err(e) => return ToolExecutionResult::failure(e),
        };

        let schema = format!("OPENBB_AGENTS.USER_{}", user.to_uppercase());

        let doc_filter = document_name
            .map(|d| format!(" AND FILE_NAME = '{}'", d.replace('\'', "''")))
            .unwrap_or_default();

        // Use AI_EMBED for semantic search with vector similarity
        let query = format!(
            r#"
            SELECT 
                FILE_NAME,
                PAGE_NUMBER,
                CHUNK_TEXT,
                VECTOR_COSINE_SIMILARITY(
                    EMBEDDING,
                    AI_EMBED('snowflake-arctic-embed-l-v2.0', '{query_text}')
                ) AS similarity
            FROM {schema}.DOCUMENT_EMBEDDINGS
            WHERE CONTENT_TYPE = 'text'{doc_filter}
            ORDER BY similarity DESC
            LIMIT {limit}
            "#,
        );

        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_get_document_images(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let document_name = match args.get("document_name").and_then(|v| v.as_str()) {
            Some(d) => d.replace('\'', "''"),
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: document_name".to_string(),
                )
            }
        };
        let page = match args.get("page").and_then(|v| v.as_i64()) {
            Some(p) => p,
            None => {
                return ToolExecutionResult::failure("Missing required argument: page".to_string())
            }
        };

        let engine_guard = engine.lock().await;
        let user = match engine_guard.get_user().await {
            Ok(u) => u,
            Err(e) => return ToolExecutionResult::failure(e),
        };

        let schema = format!("OPENBB_AGENTS.USER_{}", user.to_uppercase());

        let query = format!(
            "SELECT IMAGE_STAGE_PATH, IMAGE_HASH FROM {}.DOCUMENT_EMBEDDINGS WHERE FILE_NAME = '{}' AND PAGE_NUMBER = {} AND CONTENT_TYPE = 'image'",
            schema, document_name, page
        );

        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    async fn execute_ocr_image(
        &self,
        engine: &Arc<Mutex<SnowflakeEngine>>,
        args: &Value,
    ) -> ToolExecutionResult {
        let image_path = match args.get("image_path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: image_path".to_string(),
                )
            }
        };

        // Use Snowflake's AI_COMPLETE with vision model for OCR
        let query = format!(
            r#"
            SELECT AI_COMPLETE(
                'claude-3-5-sonnet',
                ARRAY_CONSTRUCT(
                    OBJECT_CONSTRUCT(
                        'role', 'user',
                        'content', ARRAY_CONSTRUCT(
                            OBJECT_CONSTRUCT('type', 'text', 'text', 'Extract all text from this image. Preserve the layout and structure as much as possible.'),
                            OBJECT_CONSTRUCT('type', 'image', 'source', OBJECT_CONSTRUCT('type', 'base64', 'media_type', 'image/png', 'data', BASE64_ENCODE(GET_PRESIGNED_URL('{}', 300))))
                        )
                    )
                )
            ):message:content AS ocr_text
            "#,
            image_path
        );

        let engine_guard = engine.lock().await;
        match engine_guard.execute_query(&query, None, None, None).await {
            Ok(result) => ToolExecutionResult::success(result),
            Err(e) => ToolExecutionResult::failure(e),
        }
    }

    // ========================================================================
    // Chart Tool Implementation
    // ========================================================================

    async fn execute_render_chart(&self, args: &Value) -> ToolExecutionResult {
        let chart_type = match args.get("chart_type").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => {
                return ToolExecutionResult::failure(
                    "Missing required argument: chart_type".to_string(),
                )
            }
        };
        let data = match args.get("data") {
            Some(d) => d,
            None => {
                return ToolExecutionResult::failure("Missing required argument: data".to_string())
            }
        };
        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Chart");
        let x_label = args.get("x_label").and_then(|v| v.as_str()).unwrap_or("");
        let y_label = args.get("y_label").and_then(|v| v.as_str()).unwrap_or("");

        // Generate Chart.js HTML
        let html = generate_chartjs_html(chart_type, data, title, x_label, y_label);

        // Save to temp file
        let temp_dir = std::env::temp_dir();
        let file_name = format!("chart_{}.html", uuid::Uuid::new_v4());
        let file_path = temp_dir.join(&file_name);

        match std::fs::write(&file_path, &html) {
            Ok(_) => {
                let path_str = file_path.to_string_lossy().to_string();

                // Try to open in browser
                #[cfg(target_os = "macos")]
                let _ = std::process::Command::new("open").arg(&path_str).spawn();
                #[cfg(target_os = "linux")]
                let _ = std::process::Command::new("xdg-open")
                    .arg(&path_str)
                    .spawn();
                #[cfg(target_os = "windows")]
                let _ = std::process::Command::new("cmd")
                    .args(["/C", "start", &path_str])
                    .spawn();

                ToolExecutionResult::success(json!({
                    "path": path_str,
                    "chart_type": chart_type
                }))
            }
            Err(e) => ToolExecutionResult::failure(format!("Failed to write chart file: {}", e)),
        }
    }

    // ========================================================================
    // Pagination Tool Implementation
    // ========================================================================

    async fn execute_continue_output(&self, _args: &Value) -> ToolExecutionResult {
        // This is a placeholder - actual implementation would need to maintain
        // state about truncated outputs
        ToolExecutionResult::success(json!({
            "message": "Continue output functionality requires state management"
        }))
    }
}

impl Default for ToolExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Chart.js HTML Generation
// ============================================================================

fn generate_chartjs_html(
    chart_type: &str,
    data: &Value,
    title: &str,
    x_label: &str,
    y_label: &str,
) -> String {
    let chart_type_js = match chart_type {
        "donut" => "doughnut",
        other => other,
    };

    let data_json = serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string());

    format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 800px;
            width: 90%;
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <canvas id="chart"></canvas>
    </div>
    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        const chartData = {data_json};
        
        new Chart(ctx, {{
            type: '{chart_type_js}',
            data: chartData,
            options: {{
                responsive: true,
                plugins: {{
                    title: {{
                        display: false
                    }},
                    legend: {{
                        position: 'top'
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: {x_display},
                            text: '{x_label}'
                        }}
                    }},
                    y: {{
                        title: {{
                            display: {y_display},
                            text: '{y_label}'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"#,
        title = title,
        data_json = data_json,
        chart_type_js = chart_type_js,
        x_label = x_label,
        y_label = y_label,
        x_display = if x_label.is_empty() { "false" } else { "true" },
        y_display = if y_label.is_empty() { "false" } else { "true" },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_registry_creation() {
        let registry = ToolRegistry::new();
        assert!(registry.get_tool("text2sql").is_some());
        assert!(registry.get_tool("execute_query").is_some());
        assert!(registry.get_tool("render_chart").is_some());
    }

    #[test]
    fn test_tool_categories() {
        let registry = ToolRegistry::new();
        let query_tools = registry.get_tools_by_category(ToolCategory::Query);
        assert!(!query_tools.is_empty());
    }
}
