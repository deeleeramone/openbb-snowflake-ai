use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use snowflake_connector_rs::{
    Error, SnowflakeAuthMethod, SnowflakeClient, SnowflakeClientConfig, SnowflakeSession,
};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Schema {
    #[serde(rename = "TABLE_CATALOG")]
    pub table_catalog: String,
    #[serde(rename = "TABLE_SCHEMA")]
    pub table_schema: String,
    #[serde(rename = "TABLE_NAME")]
    pub table_name: String,
    #[serde(rename = "COLUMN_NAME")]
    pub column_name: String,
    #[serde(rename = "ORDINAL_POSITION")]
    pub ordinal_position: i64,
    #[serde(rename = "COLUMN_DEFAULT")]
    pub column_default: Option<String>,
    #[serde(rename = "IS_NULLABLE")]
    pub is_nullable: String,
    #[serde(rename = "DATA_TYPE")]
    pub data_type: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SemanticModel {
    pub name: String,
    pub description: Option<String>,
    pub tables: Vec<SemanticTable>,
    pub verified_queries: Vec<VerifiedQuery>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SemanticTable {
    pub name: String,
    pub description: Option<String>,
    pub base_table: String,
    pub dimensions: Vec<SemanticColumn>,
    pub measures: Vec<SemanticMeasure>,
    pub filters: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SemanticColumn {
    pub name: String,
    pub description: Option<String>,
    pub data_type: String,
    pub expr: Option<String>,
    pub synonyms: Option<Vec<String>>,
    pub sample_values: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SemanticMeasure {
    pub name: String,
    pub description: Option<String>,
    pub expr: String,
    pub synonyms: Option<Vec<String>>,
    pub default_aggregation: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VerifiedQuery {
    pub name: String,
    pub question: String,
    pub sql: String,
    pub verified_at: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CortexMessage {
    pub role: String,
    pub content: Vec<CortexContent>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum CortexContent {
    #[serde(rename = "text")]
    Text { text: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CortexAnalystResponse {
    pub message: CortexResponseMessage,
    pub request_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CortexResponseMessage {
    pub role: String,
    pub content: Vec<CortexResponseContent>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CortexResponseContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "sql")]
    Sql { statement: String },
    #[serde(rename = "suggestions")]
    Suggestions { suggestions: Vec<String> },
}

#[derive(Clone)]
pub struct SnowflakeEngine {
    session: Arc<SnowflakeSession>,
    account: String,
    user: String,
    password: Option<String>,
    role: String,
    warehouse: String,
    database: String,
    schema: String,
    cortex_enabled: bool,
    semantic_model: Option<SemanticModel>,
    semantic_model_stage: Option<String>,
    semantic_model_file: Option<String>,
    conversation_history: Vec<CortexMessage>,
    jwt_token: Option<String>,
}

impl fmt::Debug for SnowflakeEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SnowflakeEngine")
            .field("account", &self.account)
            .field("user", &self.user)
            .field("role", &self.role)
            .field("warehouse", &self.warehouse)
            .field("database", &self.database)
            .field("schema", &self.schema)
            .field("cortex_enabled", &self.cortex_enabled)
            .finish()
    }
}

fn _get_user_schema_name(user_id: &str) -> String {
    let sanitized = user_id
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .to_uppercase();
    format!("USER_{}", sanitized)
}

impl SnowflakeEngine {
    pub fn user_schema_name(&self) -> String {
        _get_user_schema_name(&self.user)
    }

    pub async fn new(
        user: &str,
        password: &str,
        account: &str,
        role: &str,
        warehouse: &str,
        database: &str,
        schema: &str,
    ) -> Result<Self, String> {
        let client = SnowflakeClient::new(
            user,
            SnowflakeAuthMethod::Password(password.to_string()),
            SnowflakeClientConfig {
                account: account.to_string(),
                role: Some(role.to_string()),
                warehouse: Some(warehouse.to_string()),
                database: Some(database.to_string()),
                schema: Some(schema.to_string()),
                timeout: Some(std::time::Duration::from_secs(30)),
            },
        )
        .map_err(|e| e.to_string())?;
        let session = Arc::new(client.create_session().await.map_err(|e| e.to_string())?);

        let engine = SnowflakeEngine {
            session,
            account: account.to_string(),
            user: user.to_string(),
            password: Some(password.to_string()),
            role: role.to_string(),
            warehouse: warehouse.to_string(),
            database: database.to_string(),
            schema: schema.to_string(),
            cortex_enabled: true,
            semantic_model: None,
            semantic_model_stage: None,
            semantic_model_file: None,
            conversation_history: Vec::new(),
            jwt_token: None,
        };

        // Only ensure the database exists, not user-specific resources
        engine.setup_openbb_agents_database().await?;
        Ok(engine)
    }

    /// Ensures the OPENBB_AGENTS database exists. User-specific resources are created on-demand.
    async fn setup_openbb_agents_database(&self) -> Result<(), String> {
        let db_name = "OPENBB_AGENTS";
        self.execute_statement(&format!("CREATE DATABASE IF NOT EXISTS {}", db_name))
            .await?;
        Ok(())
    }

    /// Creates a user-specific schema and tables if they don't exist.
    pub async fn setup_user_resources(&self, user_id: &str) -> Result<(), String> {
        let db_name = "OPENBB_AGENTS";
        let schema_name = _get_user_schema_name(user_id);
        let fq_schema_name = format!("{}.{}", db_name, schema_name);

        // First ensure the database exists
        self.execute_statement(&format!("CREATE DATABASE IF NOT EXISTS {}", db_name))
            .await?;

        self.execute_statement(&format!("CREATE SCHEMA IF NOT EXISTS {}", fq_schema_name))
            .await?;

        self.execute_statement(&format!(
            "CREATE TABLE IF NOT EXISTS {}.AGENTS_CONVERSATIONS (
                CONVERSATION_ID STRING PRIMARY KEY,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                LAST_UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                SUMMARY STRING,
                METADATA VARIANT
            )",
            fq_schema_name
        ))
        .await?;

        self.execute_statement(&format!(
            "CREATE TABLE IF NOT EXISTS {}.AGENTS_MESSAGES (
                MESSAGE_ID STRING PRIMARY KEY,
                CONVERSATION_ID STRING,
                PARENT_MESSAGE_ID STRING,
                ROLE STRING,
                CONTENT STRING,
                METADATA VARIANT,
                TIMESTAMP TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )",
            fq_schema_name
        ))
        .await?;

        self.execute_statement(&format!(
            "CREATE TABLE IF NOT EXISTS {}.AGENTS_CONTEXT_OBJECTS (
                OBJECT_ID STRING PRIMARY KEY,
                CONVERSATION_ID STRING,
                OBJECT_TYPE STRING,
                OBJECT_KEY STRING,
                OBJECT_VALUE VARIANT,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )",
            fq_schema_name
        ))
        .await?;

        Ok(())
    }

    pub async fn add_message(
        &self,
        conversation_id: String,
        message_id: String,
        role: String,
        content: String,
    ) -> Result<(), String> {
        // Ensure user resources exist first using the actual user
        self.setup_user_resources(&self.user).await?;

        let schema_name = _get_user_schema_name(&self.user);
        let table_name = format!("OPENBB_AGENTS.{}.AGENTS_MESSAGES", schema_name);
        let query = format!(
            "INSERT INTO {} (CONVERSATION_ID, MESSAGE_ID, ROLE, CONTENT) VALUES ('{}', '{}', '{}', '{}')",
            table_name,
            conversation_id,
            message_id,
            role,
            content.replace("'", "''")
        );
        self.execute_statement(&query).await?;
        Ok(())
    }

    pub async fn get_messages(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<(String, String, String)>, String> {
        // Ensure user resources exist first using the actual user
        self.setup_user_resources(&self.user).await?;

        let schema_name = _get_user_schema_name(&self.user);
        let table_name = format!("OPENBB_AGENTS.{}.AGENTS_MESSAGES", schema_name);
        let query = format!(
            "SELECT MESSAGE_ID, ROLE, CONTENT FROM {} WHERE CONVERSATION_ID = '{}' ORDER BY TIMESTAMP ASC",
            table_name,
            conversation_id
        );
        let results = self.execute_statement(&query).await?;
        let mut messages = Vec::new();
        if let Some(rows) = results.as_array() {
            for row in rows {
                if let Some(obj) = row.as_object() {
                    let message_id = obj
                        .get("MESSAGE_ID")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let role = obj
                        .get("ROLE")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let content = obj
                        .get("CONTENT")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    messages.push((message_id, role, content));
                }
            }
        }
        Ok(messages)
    }

    pub fn generate_jwt(&mut self) -> Result<(), String> {
        // For Cortex API, we use the SNOWFLAKE_PASSWORD as the token
        if let Ok(token) = std::env::var("SNOWFLAKE_PASSWORD") {
            self.jwt_token = Some(token);
            Ok(())
        } else if let Some(pwd) = &self.password {
            self.jwt_token = Some(pwd.clone());
            Ok(())
        } else {
            Err("Failed to get token. SNOWFLAKE_PASSWORD env var not set.".to_string())
        }
    }

    pub async fn upload_semantic_model(
        &mut self,
        model_path: &PathBuf,
        stage_name: &str,
    ) -> Result<String, String> {
        let create_stage = format!(
            "CREATE STAGE IF NOT EXISTS \"{}\".\"{}\".\"{}\" \n             DIRECTORY = (ENABLE = TRUE) \n             ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')",
            self.database, self.schema, stage_name
        );
        self.session
            .query(create_stage.as_str())
            .await
            .map_err(|e| e.to_string())?;

        let model_content = std::fs::read_to_string(model_path)
            .map_err(|e| format!("Failed to read semantic model: {}", e))?;

        let semantic_model: SemanticModel = serde_yaml::from_str(&model_content)
            .or_else(|_| serde_json::from_str(&model_content))
            .map_err(|e| format!("Invalid semantic model format: {}", e))?;

        // Get the filename from path
        let filename = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("Invalid file path")?;

        let put_query = format!(
            "PUT 'file://{}' @\"{}\".\"{}\".\"{}\"/{} AUTO_COMPRESS = FALSE OVERWRITE = TRUE",
            model_path.display(),
            self.database,
            self.schema,
            stage_name,
            filename
        );
        self.session
            .query(put_query.as_str())
            .await
            .map_err(|e| e.to_string())?;

        self.semantic_model = Some(semantic_model);
        self.semantic_model_stage = Some(format!(
            "@\"{}\".\"{}\".\"{}\"/{}",
            self.database, self.schema, stage_name, filename
        ));

        Ok(format!(
            "Semantic model uploaded to {}",
            self.semantic_model_stage.as_ref().unwrap()
        ))
    }

    pub async fn verify_semantic_model(&self) -> Result<Vec<String>, String> {
        let model = self
            .semantic_model
            .as_ref()
            .ok_or("No semantic model loaded")?;
        let mut warnings = Vec::new();

        for table in &model.tables {
            let check_query = format!(
                "SELECT COUNT(*) as cnt FROM \"{}\".INFORMATION_SCHEMA.TABLES \n                 WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME = '{}'",
                self.database, self.schema, table.base_table
            );
            let results = self
                .session
                .query(check_query.as_str())
                .await
                .map_err(|e| e.to_string())?;
            let count: i64 = results[0].get("cnt").unwrap_or(0);
            if count == 0 {
                warnings.push(format!("Table {} not found", table.base_table));
            }
        }

        Ok(warnings)
    }

    pub async fn chat_with_analyst(
        &mut self,
        message: &str,
        use_history: bool,
    ) -> Result<CortexAnalystResponse, String> {
        // Check if semantic model is configured
        if self.semantic_model_stage.is_none() {
            return Err("No semantic model configured. Please upload a semantic model first or set SNOWFLAKE_FILE environment variable.".to_string());
        }

        // Generate/set token if we don't have one
        if self.jwt_token.is_none() {
            self.generate_jwt()?;
        }
        let token = self.jwt_token.as_ref().unwrap();

        let url = format!(
            "https://{}.snowflakecomputing.com/api/v2/cortex/analyst/message",
            self.account
        );

        let mut messages = if use_history {
            self.conversation_history.clone()
        } else {
            Vec::new()
        };

        let mut full_prompt = message.to_string();
        if let Ok(context) = self.get_database_context_string().await {
            if !context.is_empty() {
                full_prompt = format!("{}\n\n{}", context, message);
            }
        }

        messages.push(CortexMessage {
            role: "user".to_string(),
            content: vec![CortexContent::Text { text: full_prompt }],
        });

        let mut request_body_map = serde_json::Map::new();
        request_body_map.insert("messages".to_string(), json!(messages));

        // semantic_model_file is required by the API
        request_body_map.insert(
            "semantic_model_file".to_string(),
            json!(self.semantic_model_stage.as_ref().unwrap()),
        );

        let request_body = serde_json::Value::Object(request_body_map);

        if std::env::var("SNOWFLAKE_DEBUG").is_ok() {
            println!(
                "[CORTEX ANALYST REQUEST BODY]:\n{}",
                serde_json::to_string_pretty(&request_body).unwrap()
            );
        }

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("User-Agent", "OpenBB-Desktop-Snowflake-Client/0.1.0")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("Cortex API request failed: {}", e))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            let request_body_json = serde_json::to_string_pretty(&request_body)
                .unwrap_or_else(|e| format!("Failed to serialize request body for error: {}", e));

            if std::env::var("SNOWFLAKE_DEBUG").is_ok() {
                return Err(format!(
                    "Cortex API error {}: {}\nRequest Body: {}",
                    status, error_text, request_body_json
                ));
            } else {
                return Err(format!("Cortex API error {}: {}", status, error_text));
            }
        }

        let response_text = response
            .text()
            .await
            .map_err(|e| format!("Failed to read response: {}", e))?;

        let cortex_response: CortexAnalystResponse =
            serde_json::from_str(&response_text).map_err(|e| {
                format!(
                    "Failed to parse Cortex response: {}. Body: {}",
                    e, response_text
                )
            })?;

        if use_history {
            self.conversation_history.push(CortexMessage {
                role: "user".to_string(),
                content: vec![CortexContent::Text {
                    text: message.to_string(),
                }],
            });

            self.conversation_history.push(CortexMessage {
                role: cortex_response.message.role.clone(),
                content: cortex_response
                    .message
                    .content
                    .iter()
                    .map(|c| match c {
                        CortexResponseContent::Text { text } => {
                            CortexContent::Text { text: text.clone() }
                        }
                        CortexResponseContent::Sql { statement } => CortexContent::Text {
                            text: statement.clone(),
                        },
                        CortexResponseContent::Suggestions { suggestions } => CortexContent::Text {
                            text: suggestions.join(", "),
                        },
                    })
                    .collect(),
            });
        }

        Ok(cortex_response)
    }

    pub fn reset_conversation(&mut self) {
        self.conversation_history.clear();
    }

    pub fn get_conversation_history(&self) -> &[CortexMessage] {
        &self.conversation_history
    }

    pub async fn get_database_context_string(&self) -> Result<String, String> {
        let tables = self.get_table_list().await?;
        if tables.is_empty() {
            return Ok(String::new());
        }

        let mut context = format!(
            "-- Current database: {}, schema: {}\n-- Tables:\n",
            self.database, self.schema
        );

        for table_name in tables {
            match self.get_table_info(&table_name).await {
                Ok(schema_json_str) => {
                    let schema_json: serde_json::Value =
                        match serde_json::from_str(&schema_json_str) {
                            Ok(json) => json,
                            Err(_) => continue, // Skip if JSON is invalid
                        };

                    if let Some(columns_val) = schema_json.get("columns") {
                        if let Some(columns_arr) = columns_val.as_array() {
                            if !columns_arr.is_empty() {
                                let full_table_name =
                                    format!("{}.{}.{}", self.database, self.schema, table_name);
                                let column_names: Vec<String> = columns_arr
                                    .iter()
                                    .filter_map(|c| {
                                        c.get("COLUMN_NAME")
                                            .and_then(|n| n.as_str())
                                            .map(String::from)
                                    })
                                    .collect();
                                context.push_str(&format!(
                                    "-- {}({}\n",
                                    full_table_name,
                                    column_names.join(", ")
                                ));
                            }
                        }
                    }
                }
                Err(_) => continue, // Skip tables we can't get info for
            }
        }
        Ok(context)
    }

    pub async fn configure_from_env(&mut self) {
        // Get the semantic model filename and stage from environment
        let file_name = match std::env::var("SNOWFLAKE_FILE") {
            Ok(name) if !name.is_empty() => name,
            _ => return,
        };

        let stage_name =
            std::env::var("SNOWFLAKE_STAGE").unwrap_or_else(|_| "SEMANTIC_MODELS".to_string());

        // Construct the stage path: @DATABASE.SCHEMA.STAGE/filename
        let stage_path = format!(
            "@\"{}\".\"{}\".\"{}\"/{}",
            self.database, self.schema, stage_name, file_name
        );

        // Set the semantic model stage path directly
        self.semantic_model_stage = Some(stage_path);
    }

    pub async fn execute_with_cortex_context(
        &mut self, // Changed to &mut
        question: &str,
        execute: bool,
    ) -> Result<serde_json::Value, String> {
        let response = self.chat_with_analyst(question, false).await?;

        let mut sql = None;
        let mut explanation = None;

        for content in response.message.content {
            match content {
                CortexResponseContent::Sql { statement } => sql = Some(statement),
                CortexResponseContent::Text { text } => explanation = Some(text),
                _ => {} // Ignore other content types for now
            }
        }

        let sql = sql.ok_or("No SQL generated")?;

        if execute {
            let results = self.execute_query(&sql, None, None, None).await?;
            Ok(json!({
                "sql": sql,
                "explanation": explanation,
                "results": results,
                "request_id": response.request_id,
            }))
        } else {
            Ok(json!({
                "sql": sql,
                "explanation": explanation,
                "request_id": response.request_id,
            }))
        }
    }

    pub async fn generate_semantic_model_from_schema(&self) -> Result<SemanticModel, String> {
        let schema = self.get_detailed_schema().await?;
        let mut tables_map: std::collections::HashMap<String, Vec<Schema>> =
            std::collections::HashMap::new();

        for col in schema {
            tables_map
                .entry(col.table_name.clone())
                .or_default()
                .push(col);
        }

        let mut semantic_tables = Vec::new();
        for (table_name, columns) in tables_map {
            let dimensions: Vec<SemanticColumn> = columns
                .iter()
                .map(|col| SemanticColumn {
                    name: col.column_name.clone(),
                    description: Some(format!("{} column from {}", col.column_name, table_name)),
                    data_type: col.data_type.clone(),
                    expr: None,
                    synonyms: None,
                    sample_values: None,
                })
                .collect();

            semantic_tables.push(SemanticTable {
                name: table_name.clone(),
                description: Some(format!("Auto-generated semantic model for {}", table_name)),
                base_table: format!(
                    "\"{}\".\"{}\".\"{}\"",
                    self.database, self.schema, table_name
                ),
                dimensions,
                measures: vec![],
                filters: None,
            });
        }

        Ok(SemanticModel {
            name: format!("{}_semantic_model", self.schema),
            description: Some("Auto-generated semantic model".to_string()),
            tables: semantic_tables,
            verified_queries: vec![],
        })
    }

    pub async fn validate_query(&self, query: &str) -> Result<(), String> {
        let explain_query = format!("EXPLAIN {}", query);
        self.session
            .query(explain_query.as_str())
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn get_query_suggestions(
        &mut self,
        partial_query: &str,
    ) -> Result<Vec<String>, String> {
        if !self.cortex_enabled {
            return Ok(vec![]);
        }

        let mut full_prompt = partial_query.to_string();
        if let Ok(context) = self.get_database_context_string().await {
            if !context.is_empty() {
                full_prompt = format!("{}\n\nComplete this SQL query, ensuring all table names are fully qualified (e.g., DATABASE.SCHEMA.TABLE): {}", context, partial_query);
            }
        }

        match self
            .chat_with_analyst(
                &full_prompt,
                false, // Don't use history for suggestions
            )
            .await
        {
            Ok(response) => {
                if let Some(content) = response.message.content.into_iter().next() {
                    match content {
                        CortexResponseContent::Suggestions { suggestions } => {
                            return Ok(suggestions)
                        }
                        CortexResponseContent::Text { text } => {
                            let suggestions: Vec<String> = text
                                .lines()
                                .filter(|line| !line.trim().is_empty())
                                .map(|line| line.trim().to_string())
                                .collect();
                            return Ok(suggestions);
                        }
                        CortexResponseContent::Sql { statement } => {
                            return Ok(vec![statement]);
                        }
                    }
                }
                Ok(vec![])
            }
            Err(_) => Ok(vec![]), // Silently fail for suggestions
        }
    }

    fn build_where_clause(filter_model: &serde_json::Value) -> String {
        let mut where_clauses = Vec::new();
        if let Some(map) = filter_model.as_object() {
            for (column, filter) in map {
                if let Some(filter_type) = filter.get("filterType").and_then(|v| v.as_str()) {
                    match filter_type {
                        "text" => {
                            if let Some(filter_value) =
                                filter.get("filter").and_then(|v| v.as_str())
                            {
                                where_clauses.push(format!("{} LIKE '%{}%'", column, filter_value));
                            }
                        }
                        "number" => {
                            if let Some(filter_value) =
                                filter.get("filter").and_then(|v| v.as_i64())
                            {
                                where_clauses.push(format!("{} = {}", column, filter_value));
                            }
                        }
                        "date" => {
                            if let (Some(date_from), Some(date_to)) = (
                                filter.get("dateFrom").and_then(|v| v.as_str()),
                                filter.get("dateTo").and_then(|v| v.as_str()),
                            ) {
                                where_clauses.push(format!(
                                    "{} BETWEEN '{}' AND '{}'",
                                    column, date_from, date_to
                                ));
                            }
                        }
                        _ => {} // Ignore unknown filter types
                    }
                }
            }
        }
        if where_clauses.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", where_clauses.join(" AND "))
        }
    }

    pub async fn execute_statement(&self, statement: &str) -> Result<serde_json::Value, String> {
        let mut cleaned_query = statement.trim().to_string();
        if cleaned_query.ends_with(';') {
            cleaned_query.pop();
        }

        let results = self
            .session
            .query(cleaned_query.as_str())
            .await
            .map_err(|e| e.to_string())?;

        let column_names: Vec<String> = results.first().map_or(vec![], |first_row| {
            first_row
                .column_types()
                .into_iter()
                .map(|col| col.name().to_string())
                .collect()
        });

        let row_data: Vec<_> = results
            .into_iter()
            .map(|row| {
                let mut row_map = serde_json::Map::new();
                for name in &column_names {
                    let value = match row.get::<serde_json::Value>(name) {
                        Ok(v) => v,
                        Err(_) => match row.get::<String>(name) {
                            Ok(s) => serde_json::Value::String(s),
                            Err(_) => serde_json::Value::Null,
                        },
                    };
                    row_map.insert(name.clone(), value);
                }
                serde_json::Value::Object(row_map)
            })
            .collect();

        Ok(json!(row_data))
    }

    pub async fn execute_query(
        &self,
        query: &str,
        start_row: Option<usize>,
        end_row: Option<usize>,
        filter_model: Option<&serde_json::Value>,
    ) -> Result<serde_json::Value, String> {
        let mut cleaned_query = query.trim().to_string();
        if cleaned_query.ends_with(';') {
            cleaned_query.pop();
        }

        let where_clause = filter_model.map_or(String::new(), Self::build_where_clause);
        let filtered_query = if where_clause.is_empty() {
            cleaned_query
        } else {
            format!("{} {}", cleaned_query, where_clause)
        };

        let count_query = format!("SELECT COUNT(*) as count FROM ({})", filtered_query);
        let count_results = self
            .session
            .query(count_query.as_str())
            .await
            .map_err(|e| e.to_string())?;
        let row_count: i64 = count_results[0].get("count").unwrap_or(0);

        let paginated_query = if let (Some(start), Some(end)) = (start_row, end_row) {
            format!("{} LIMIT {} OFFSET {}", filtered_query, end - start, start)
        } else {
            filtered_query
        };
        let results = self
            .session
            .query(paginated_query.as_str())
            .await
            .map_err(|e| e.to_string())?;

        let column_defs: Vec<_> = results.first().map_or(vec![], |first_row| {
            first_row
                .column_types()
                .into_iter()
                .map(|col| {
                    json!({
                        "field": col.name(),
                        "type": col.column_type().snowflake_type(),
                    })
                })
                .collect()
        });

        let column_names: Vec<String> = column_defs
            .iter()
            .map(|def| def["field"].as_str().unwrap_or_default().to_string())
            .collect();

        let row_data: Vec<_> = results
            .into_iter()
            .map(|row| {
                let mut row_map = serde_json::Map::new();
                for name in &column_names {
                    let value = match row.get::<serde_json::Value>(name) {
                        Ok(v) => v,
                        Err(_) => match row.get::<String>(name) {
                            Ok(s) => serde_json::Value::String(s),
                            Err(_) => serde_json::Value::Null,
                        },
                    };
                    row_map.insert(name.clone(), value);
                }
                serde_json::Value::Object(row_map)
            })
            .collect();

        Ok(json!({
            "columnDefs": column_defs,
            "rowData": row_data,
            "rowCount": if start_row.is_some() && end_row.is_some() { 1 } else { row_count },
        }))
    }

    pub async fn get_detailed_schema(&self) -> Result<Vec<Schema>, String> {
        let query = format!(
            "SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, \n             ORDINAL_POSITION, COLUMN_DEFAULT, IS_NULLABLE, DATA_TYPE \n             FROM \"{}\".INFORMATION_SCHEMA.COLUMNS \n             WHERE TABLE_SCHEMA = '{}' \n             ORDER BY TABLE_NAME, ORDINAL_POSITION",
            self.database, self.schema
        );

        let results = self
            .session
            .query(query.as_str())
            .await
            .map_err(|e| e.to_string())?;
        let mut schema = Vec::new();

        for row in results {
            schema.push(Schema {
                table_catalog: row.get("TABLE_CATALOG").map_err(|e| e.to_string())?,
                table_schema: row.get("TABLE_SCHEMA").map_err(|e| e.to_string())?,
                table_name: row.get("TABLE_NAME").map_err(|e| e.to_string())?,
                column_name: row.get("COLUMN_NAME").map_err(|e| e.to_string())?,
                ordinal_position: row.get("ORDINAL_POSITION").map_err(|e| e.to_string())?,
                column_default: row.get("COLUMN_DEFAULT").ok(),
                is_nullable: row.get("IS_NULLABLE").map_err(|e| e.to_string())?,
                data_type: row.get("DATA_TYPE").map_err(|e| e.to_string())?,
            });
        }

        Ok(schema)
    }

    pub async fn get_table_info(&self, table_name: &str) -> Result<String, String> {
        let parts: Vec<&str> = table_name.split('.').collect();
        let (catalog, schema, table) = match parts.as_slice() {
            [c, s, t] => (*c, *s, *t),
            [s, t] => (self.database.as_str(), *s, *t),
            [t] => (self.database.as_str(), self.schema.as_str(), *t),
            _ => return Err("Invalid table name format".to_string()),
        };

        // Always query directly from Snowflake (no cache)
        let full_table_name = format!("\"{}\".\"{}\".\"{}\"", catalog, schema, table);
        let describe_query = format!("DESCRIBE TABLE {}", full_table_name);

        let describe_results = self
            .session
            .query(describe_query.as_str())
            .await
            .map_err(|e| format!("Failed to describe table: {}", e))?;

        // Convert each row to JSON with ALL metadata columns
        let columns_json: Vec<serde_json::Value> = describe_results
            .into_iter()
            .filter_map(|row| {
                let mut col = serde_json::Map::new();
                col.insert(
                    "COLUMN_NAME".to_string(),
                    json!(row.get::<String>("name").ok()?),
                );
                col.insert(
                    "DATA_TYPE".to_string(),
                    json!(row.get::<String>("type").ok()?),
                );
                col.insert(
                    "IS_NULLABLE".to_string(),
                    json!(row.get::<String>("null?").unwrap_or_default()),
                );
                col.insert(
                    "COLUMN_DEFAULT".to_string(),
                    json!(row.get::<Option<String>>("default").ok().flatten()),
                );
                col.insert("ORDINAL_POSITION".to_string(), json!(0)); // Not provided by DESCRIBE
                Some(serde_json::Value::Object(col))
            })
            .collect();

        Ok(serde_json::to_string(&json!({
            "columns": columns_json,
            "primary_keys": [],
            "foreign_keys": [],
        }))
        .unwrap())
    }

    /// Get list of tables in current schema
    pub async fn get_table_list(&self) -> Result<Vec<String>, String> {
        // Always query directly from Snowflake
        self.list_tables_in_current_schema_direct()
            .await
            .map_err(|e| e.to_string())
    }

    async fn list_tables_in_current_schema_direct(&self) -> Result<Vec<String>, Error> {
        let query = format!(
            "SELECT TABLE_NAME FROM \"{}\".INFORMATION_SCHEMA.TABLES              WHERE TABLE_SCHEMA = '{}'              ORDER BY TABLE_NAME",
            self.database, self.schema
        );

        let results = self.session.query(query.as_str()).await?;
        let mut tables = Vec::new();
        for row in results {
            if let Ok(name) = row.get::<String>("TABLE_NAME") {
                tables.push(name);
            }
        }
        Ok(tables)
    }

    pub async fn get_available_models(&self) -> Result<Vec<(String, String)>, String> {
        let models = vec![
            ("claude-sonnet-4-5".to_string(), "Claude Sonnet 4.5 (Preview)".to_string()),
            ("claude-haiku-4-5".to_string(), "Claude Haiku 4.5 (Preview)".to_string()),
            ("claude-4-sonnet".to_string(), "Claude 4 Sonnet".to_string()),
            ("claude-4-opus".to_string(), "Claude 4 Opus".to_string()),
            ("claude-3-7-sonnet".to_string(), "Claude 3.7 Sonnet".to_string()),
            ("claude-3-5-sonnet".to_string(), "Claude 3.5 Sonnet".to_string()),
            ("openai-gpt-4.1".to_string(), "OpenAI GPT-4.1".to_string()),
            ("openai-o4-mini".to_string(), "OpenAI o4-mini".to_string()),
            ("openai-gpt-5".to_string(), "OpenAI GPT-5 (Preview)".to_string()),
            ("openai-gpt-5-mini".to_string(), "OpenAI GPT-5 Mini (Preview)".to_string()),
            ("openai-gpt-5-nano".to_string(), "OpenAI GPT-5 Nano (Preview)".to_string()),
            ("openai-gpt-5-chat".to_string(), "OpenAI GPT-5 Chat".to_string()),
            ("openai-gpt-oss-120b".to_string(), "OpenAI GPT OSS 120B (Preview)".to_string()),
            ("llama4-maverick".to_string(), "Llama 4 Maverick".to_string()),
            ("llama3.1-8b".to_string(), "Llama 3.1 8B".to_string()),
            ("llama3.1-70b".to_string(), "Llama 3.1 70B".to_string()),
            ("llama3.1-405b".to_string(), "Llama 3.1 405B".to_string()),
            ("deepseek-r1".to_string(), "DeepSeek R1".to_string()),
            ("mistral-7b".to_string(), "Mistral 7B".to_string()),
            ("mistral-large".to_string(), "Mistral Large".to_string()),
            ("mistral-large2".to_string(), "Mistral Large 2".to_string()),
            ("snowflake-llama-3.3-70b".to_string(), "Snowflake Llama 3.3 70B".to_string()),
        ];
        Ok(models)
    }

    /// List documents available in the user's Cortex uploads stage
    /// Returns a list of (file_name, stage_path, is_parsed, page_count) tuples
    /// This lists files from the stage and checks if they have been parsed
    pub async fn list_cortex_documents(&self) -> Result<Vec<(String, String, bool, i64)>, String> {
        let schema_name = _get_user_schema_name(&self.user);
        let stage_name = format!("OPENBB_AGENTS.{}.CORTEX_UPLOADS", schema_name);
        let table_name = format!("OPENBB_AGENTS.{}.DOCUMENT_PARSE_RESULTS", schema_name);

        // First, list files from the stage
        let list_query = format!(
            "LIST @\"OPENBB_AGENTS\".\"{}\".\"CORTEX_UPLOADS\"",
            schema_name
        );

        let stage_files = match self.session.query(list_query.as_str()).await {
            Ok(results) => {
                let mut files = Vec::new();
                for row in results {
                    if let Ok(name) = row.get::<String>("name") {
                        // Extract just the filename from the full path
                        let filename = name.split('/').next_back().unwrap_or(&name).to_string();
                        if !filename.is_empty() && !filename.ends_with('/') {
                            files.push(filename);
                        }
                    }
                }
                files
            }
            Err(_) => Vec::new(), // Stage might not exist
        };

        if stage_files.is_empty() {
            return Ok(Vec::new());
        }

        // Check which files have been parsed and get their page counts
        let mut documents = Vec::new();

        for file_name in stage_files {
            let stage_path = format!("@{}/{}", stage_name, file_name);

            // Check if this file has been parsed
            let check_query = format!(
                "SELECT COUNT(*) as PAGE_COUNT FROM \"OPENBB_AGENTS\".\"{}\".\"DOCUMENT_PARSE_RESULTS\" WHERE FILE_NAME = '{}'",
                schema_name,
                file_name.replace("'", "''")
            );

            let (is_parsed, page_count) = match self.session.query(check_query.as_str()).await {
                Ok(results) => {
                    if let Some(row) = results.into_iter().next() {
                        let count: i64 = row.get("PAGE_COUNT").unwrap_or(0);
                        (count > 0, count)
                    } else {
                        (false, 0)
                    }
                }
                Err(_) => (false, 0), // Table might not exist
            };

            documents.push((file_name, stage_path, is_parsed, page_count));
        }

        Ok(documents)
    }

    /// Download a document from Snowflake stage and return base64-encoded content
    /// document_id is the FILE_NAME - file is downloaded from CORTEX_UPLOADS stage
    pub async fn download_cortex_document(&self, document_id: &str) -> Result<String, String> {
        let schema_name = _get_user_schema_name(&self.user);

        // Extract just the filename if a full stage path was provided
        // Input could be: "@OPENBB_AGENTS.USER_XXX.CORTEX_UPLOADS/file.pdf" or just "file.pdf"
        let file_name = if document_id.contains('/') {
            // Extract filename from path like "@OPENBB_AGENTS.USER_XXX.CORTEX_UPLOADS/file.pdf"
            document_id.rsplit('/').next().unwrap_or(document_id)
        } else {
            document_id
        };

        // Use GET command to download the file via Snow CLI
        let snow_cli_path = self.find_snow_cli().ok_or_else(|| {
            "Snow CLI not found. Please install snowflake-cli-python.".to_string()
        })?;

        // Create a temp directory for the download
        let temp_dir =
            std::env::temp_dir().join(format!("snowflake_download_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| format!("Failed to create temp directory: {}", e))?;

        // Build GET query - proper syntax per Snowflake docs:
        // GET @[namespace.]stage_name/path 'file:///local_path/'
        // The stage path should be: '@OPENBB_AGENTS.USER_XXX.CORTEX_UPLOADS/filename'
        // Local path needs trailing slash for directory
        // Both paths must be single-quoted, especially if filename contains spaces
        let local_path = format!("file://{}/", temp_dir.to_string_lossy().replace("\\", "/"));
        let stage_path = format!(
            "@OPENBB_AGENTS.{}.CORTEX_UPLOADS/{}",
            schema_name, file_name
        );

        let get_query = format!("GET '{}' '{}'", stage_path, local_path);

        eprintln!("[DEBUG] GET query: {}", get_query);
        eprintln!("[DEBUG] Temp dir: {}", temp_dir.display());

        let mut cmd = std::process::Command::new(&snow_cli_path);
        cmd.arg("sql")
            .arg("-q")
            .arg(&get_query)
            .arg("--database")
            .arg("OPENBB_AGENTS")
            .arg("--schema")
            .arg(&schema_name);

        if let Some(pwd) = &self.password {
            cmd.env("SNOWFLAKE_PASSWORD", pwd);
        }

        let output = cmd
            .output()
            .map_err(|e| format!("Failed to execute Snow CLI: {}", e))?;

        let stdout_str = String::from_utf8_lossy(&output.stdout);
        let stderr_str = String::from_utf8_lossy(&output.stderr);
        eprintln!("[DEBUG] Snow CLI stdout: {}", stdout_str);
        eprintln!("[DEBUG] Snow CLI stderr: {}", stderr_str);

        if !output.status.success() {
            // Clean up temp dir
            let _ = std::fs::remove_dir_all(&temp_dir);
            return Err(format!("Failed to download file: {}", stderr_str));
        }

        // List files in temp directory to see what was downloaded
        let mut found_file: Option<std::path::PathBuf> = None;
        if let Ok(entries) = std::fs::read_dir(&temp_dir) {
            eprintln!("[DEBUG] Files in temp dir:");
            for entry in entries.flatten() {
                let path = entry.path();
                eprintln!("[DEBUG]   - {}", path.display());
                // Snowflake may compress files with .gz extension
                let entry_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if entry_name == file_name || entry_name == format!("{}.gz", file_name) {
                    found_file = Some(path);
                }
            }
        }

        // Try the exact filename first, then check for .gz version
        let downloaded_file = if let Some(f) = found_file {
            f
        } else {
            temp_dir.join(file_name)
        };

        eprintln!("[DEBUG] Looking for file at: {}", downloaded_file.display());

        // Check if file exists
        if !downloaded_file.exists() {
            let _ = std::fs::remove_dir_all(&temp_dir);
            return Err(format!(
                "Downloaded file not found at '{}'. GET command output: {}",
                downloaded_file.display(),
                stdout_str
            ));
        }

        let file_content = if downloaded_file.extension().and_then(|e| e.to_str()) == Some("gz") {
            // Decompress gzip file
            use std::io::Read;
            let file = std::fs::File::open(&downloaded_file)
                .map_err(|e| format!("Failed to open compressed file: {}", e))?;
            let mut decoder = flate2::read::GzDecoder::new(file);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| format!("Failed to decompress file: {}", e))?;
            decompressed
        } else {
            std::fs::read(&downloaded_file).map_err(|e| {
                format!(
                    "Failed to read downloaded file '{}': {}",
                    downloaded_file.display(),
                    e
                )
            })?
        };

        eprintln!("[DEBUG] Read {} bytes from file", file_content.len());

        // Clean up temp directory
        let _ = std::fs::remove_dir_all(&temp_dir);

        // Return base64-encoded content for binary files
        use base64::Engine;
        Ok(base64::engine::general_purpose::STANDARD.encode(&file_content))
    }

    /// Upload bytes directly to a Snowflake stage (for widget PDF uploads)
    /// Returns the stage path where the file was uploaded
    /// file_name can include subdirectories (e.g., "doc.pdf/page_1_image_0.jpeg")
    pub async fn upload_bytes_to_stage(
        &self,
        file_bytes: &[u8],
        file_name: &str,
        stage_name: Option<&str>,
    ) -> Result<String, String> {
        let schema_name = _get_user_schema_name(&self.user);
        let stage = stage_name.unwrap_or("CORTEX_UPLOADS");
        let qualified_stage_name = format!("OPENBB_AGENTS.{}.{}", schema_name, stage);

        // Ensure stage exists
        let create_stage_query = format!(
            "CREATE STAGE IF NOT EXISTS \"OPENBB_AGENTS\".\"{}\".\"{}\" \
             DIRECTORY = (ENABLE = TRUE) \
             ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')",
            schema_name, stage
        );
        self.session
            .query(create_stage_query.as_str())
            .await
            .map_err(|e| format!("Failed to create stage: {}", e))?;

        // Handle subdirectories in file_name (e.g., "doc.pdf/page_1_image_0.jpeg")
        // Extract just the base filename for the temp file, and the subdirectory for the stage path
        let path = std::path::Path::new(file_name);
        let base_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(file_name);
        let subdir = path.parent().and_then(|p| p.to_str()).unwrap_or("");

        // Write bytes to temp file (just the base filename)
        let temp_dir =
            std::env::temp_dir().join(format!("snowflake_upload_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| format!("Failed to create temp directory: {}", e))?;

        let temp_file_path = temp_dir.join(base_name);
        std::fs::write(&temp_file_path, file_bytes)
            .map_err(|e| format!("Failed to write temp file: {}", e))?;

        // Find Snow CLI
        let snow_cli_path = self.find_snow_cli().ok_or_else(|| {
            "Snow CLI not found. Please install snowflake-cli-python.".to_string()
        })?;

        // Build PUT query - include subdirectory in stage path if present
        let temp_file_str = temp_file_path.to_string_lossy().replace("\\", "/");
        let stage_target = if subdir.is_empty() {
            format!("@\"{}\".\"{}\"", schema_name, stage)
        } else {
            // PUT to subdirectory: @SCHEMA.STAGE/subdir/
            format!("@\"{}\".\"{}\"/{}/", schema_name, stage, subdir)
        };
        let put_query = format!(
            "PUT 'file://{}' {} AUTO_COMPRESS = FALSE OVERWRITE = TRUE",
            temp_file_str, stage_target
        );

        let mut cmd = std::process::Command::new(&snow_cli_path);
        cmd.arg("sql")
            .arg("-q")
            .arg(&put_query)
            .arg("--database")
            .arg("OPENBB_AGENTS")
            .arg("--schema")
            .arg(&schema_name);

        if let Some(pwd) = &self.password {
            cmd.env("SNOWFLAKE_PASSWORD", pwd);
        }

        let output = cmd
            .output()
            .map_err(|e| format!("Failed to execute Snow CLI: {}", e))?;

        // Clean up temp directory
        let _ = std::fs::remove_dir_all(&temp_dir);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Failed to upload file: {}", stderr));
        }

        // Refresh stage directory
        let refresh_query = format!(
            "ALTER STAGE \"OPENBB_AGENTS\".\"{}\".\"{}\" REFRESH",
            schema_name, stage
        );
        let _ = self.session.query(refresh_query.as_str()).await;

        // Return the stage path with full path including subdirectory
        Ok(format!("@{}/{}", qualified_stage_name, file_name))
    }

    /// List all stages available to the user
    pub async fn list_stages(&self) -> Result<Vec<String>, String> {
        // Always query directly from Snowflake
        self.list_stages_direct().await.map_err(|e| e.to_string())
    }

    /// List files in a specific stage
    pub async fn list_files_in_stage(
        &self,
        stage_name: &str,
        _no_cache: bool,
    ) -> Result<Vec<String>, String> {
        // Always query directly from Snowflake (ignore no_cache parameter)
        self.list_files_in_stage_direct(stage_name)
            .await
            .map_err(|e| e.to_string())
    }

    async fn list_stages_direct(&self) -> Result<Vec<String>, Error> {
        let results = self.session.query("SHOW STAGES").await?;

        let mut stages = Vec::new();
        for row in results {
            if let Ok(name) = row.get::<String>("name") {
                stages.push(name);
            }
        }
        Ok(stages)
    }

    async fn list_files_in_stage_direct(&self, stage_name: &str) -> Result<Vec<String>, Error> {
        let query = format!("LIST @\"{}\"", stage_name);
        let results = self.session.query(query.as_str()).await?;

        let mut files = Vec::new();
        for row in results {
            if let Ok(name) = row.get::<String>("name") {
                // Extract just the filename from the full path
                let filename = name.split('/').next_back().unwrap_or(&name);
                files.push(filename.to_string());
            }
        }
        Ok(files)
    }

    /// List all databases accessible to the user
    pub async fn list_databases(&self) -> Result<Vec<String>, String> {
        let query = "SHOW DATABASES";
        let results = self.session.query(query).await.map_err(|e| e.to_string())?;

        let mut databases = Vec::new();
        for row in results {
            if let Ok(name) = row.get::<String>("name") {
                databases.push(name);
            }
        }
        Ok(databases)
    }

    /// List all schemas in a specific database (or current if None)
    pub async fn list_schemas(&self, database: Option<&str>) -> Result<Vec<String>, String> {
        let db_to_use = database.unwrap_or(&self.database);
        // Always query directly from Snowflake
        self.list_schemas_direct(Some(db_to_use))
            .await
            .map_err(|e| e.to_string())
    }

    async fn list_schemas_direct(&self, database: Option<&str>) -> Result<Vec<String>, Error> {
        let query = if let Some(db) = database {
            format!("SHOW SCHEMAS IN DATABASE \"{}\"", db)
        } else {
            "SHOW SCHEMAS".to_string()
        };

        let results = self.session.query(query.as_str()).await?;

        let mut schemas = Vec::new();
        for row in results {
            if let Ok(name) = row.get::<String>("name") {
                schemas.push(name);
            }
        }
        Ok(schemas)
    }

    /// List all warehouses accessible to the user
    pub async fn list_warehouses(&self) -> Result<Vec<String>, String> {
        // Always query directly from Snowflake
        self.list_warehouses_direct()
            .await
            .map_err(|e| e.to_string())
    }

    async fn list_warehouses_direct(&self) -> Result<Vec<String>, Error> {
        let query = "SHOW WAREHOUSES";
        let results = self.session.query(query).await?;

        let mut warehouses = Vec::new();
        for row in results {
            if let Ok(name) = row.get::<String>("name") {
                warehouses.push(name);
            }
        }
        Ok(warehouses)
    }

    /// Switch to a different database
    pub async fn use_database(&mut self, database: &str) -> Result<(), Error> {
        let query = format!("USE DATABASE \"{}\"", database);
        self.session.query(query.as_str()).await?;
        self.database = database.to_string();
        Ok(())
    }

    /// Switch to a different schema
    pub async fn use_schema(&mut self, schema: &str) -> Result<(), Error> {
        let query = format!("USE SCHEMA \"{}\"", schema);
        self.session.query(query.as_str()).await?;
        self.schema = schema.to_string();
        Ok(())
    }

    /// Switch to a different warehouse
    pub async fn use_warehouse(&mut self, warehouse: &str) -> Result<(), Error> {
        let query = format!("USE WAREHOUSE \"{}\"", warehouse);
        self.session.query(query.as_str()).await?;
        Ok(())
    }

    /// Switch database and schema together
    pub async fn use_database_schema(&mut self, database: &str, schema: &str) -> Result<(), Error> {
        self.use_database(database).await?;
        self.use_schema(schema).await?;
        Ok(())
    }

    /// List all tables across all accessible databases and schemas
    pub async fn list_all_tables(&self) -> Result<Vec<(String, String, String)>, String> {
        // Always query directly from Snowflake
        // This might be slow, so consider limiting scope
        self.list_all_tables_direct()
            .await
            .map_err(|e| e.to_string())
    }

    /// List tables in a specific database and schema
    pub async fn list_tables_in(
        &self,
        database: &str,
        schema: &str,
    ) -> Result<Vec<String>, String> {
        // Always query directly from Snowflake
        self.list_tables_in_direct(database, schema)
            .await
            .map_err(|e| e.to_string())
    }

    async fn list_tables_in_direct(
        &self,
        database: &str,
        schema: &str,
    ) -> Result<Vec<String>, Error> {
        let query = format!(
            "SELECT TABLE_NAME FROM \"{}\".INFORMATION_SCHEMA.TABLES              WHERE TABLE_SCHEMA = '{}'              ORDER BY TABLE_NAME",
            database, schema
        );

        let results = self.session.query(query.as_str()).await?;
        let mut tables = Vec::new();
        for row in results {
            if let Ok(name) = row.get::<String>("TABLE_NAME") {
                tables.push(name);
            }
        }
        Ok(tables)
    }

    /// List all tables across all accessible databases and schemas
    async fn list_all_tables_direct(&self) -> Result<Vec<(String, String, String)>, String> {
        let databases = self
            .list_databases()
            .await
            .map_err(|e| format!("Failed to list databases: {}", e))?;
        let mut all_tables = Vec::new();

        for db in &databases {
            match self
                .list_schemas_direct(Some(db))
                .await
                .map_err(|e| e.to_string())
            {
                Ok(schemas) => {
                    for schema in &schemas {
                        let query = format!(
                            "SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME \
                             FROM \"{}\".INFORMATION_SCHEMA.TABLES \
                             WHERE TABLE_SCHEMA = '{}' \
                             ORDER BY TABLE_NAME",
                            db, schema
                        );

                        match self
                            .session
                            .query(query.as_str())
                            .await
                            .map_err(|e| e.to_string())
                        {
                            Ok(results) => {
                                for row in results {
                                    if let (Ok(catalog), Ok(schema_name), Ok(table)) = (
                                        row.get::<String>("TABLE_CATALOG"),
                                        row.get::<String>("TABLE_SCHEMA"),
                                        row.get::<String>("TABLE_NAME"),
                                    ) {
                                        all_tables.push((catalog, schema_name, table));
                                    }
                                }
                            }
                            Err(_) => continue,
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        Ok(all_tables)
    }

    pub async fn close(&mut self) -> Result<(), String> {
        // Nothing to close anymore (no cache)
        Ok(())
    }

    fn find_snow_cli(&self) -> Option<PathBuf> {
        if let Ok(path) = std::env::var("PATH") {
            for part in std::env::split_paths(&path) {
                let exe_name = if cfg!(windows) { "snow.exe" } else { "snow" };
                let exe = part.join(exe_name);
                if exe.exists() && !exe.is_dir() {
                    return Some(exe);
                }
            }
        }

        // On Windows, which("snow") should find snow.cmd
        if let Some(path) = which::which("snow").ok() {
            return Some(path);
        }

        None
    }

    pub async fn upload_file_to_stage(
        &self,
        file_path: &Path,
        stage_name: &str,
    ) -> Result<String, String> {
        // First ensure the stage exists
        let qualified_stage_name = format!(
            "\"{}\".\"{}\".\"{}\"",
            self.database, self.schema, stage_name
        );

        let create_stage_query = format!(
            "CREATE STAGE IF NOT EXISTS {} DIRECTORY = (ENABLE = TRUE) ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')",
            qualified_stage_name
        );

        self.session
            .query(create_stage_query.as_str())
            .await
            .map_err(|e| format!("Failed to create stage: {}", e))?;

        // Refresh directory table
        let refresh_stage_query = format!("ALTER STAGE {} REFRESH", qualified_stage_name);
        self.session
            .query(refresh_stage_query.as_str())
            .await
            .map_err(|e| format!("Failed to refresh stage: {}", e))?;

        // Check if file exists
        if !file_path.exists() {
            return Err(format!("File does not exist: {}", file_path.display()));
        }

        // Find Snow CLI
        let snow_cli_path = self
            .find_snow_cli()
            .ok_or_else(|| "Snow CLI binary not found. Please install `snowflake-cli-python` and ensure it's in your PATH.".to_string())?;

        // Use absolute path for the file
        let abs_path = if file_path.is_absolute() {
            file_path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|e| e.to_string())? // Use map_err for better error handling
                .join(file_path)
        };
        // Normalize path separators for Windows to forward slashes as required by PUT command
        let abs_path_str = abs_path.to_string_lossy().replace("\\", "/");

        // Determine auto-compression
        let compressed_extensions = ["gz", "bz2", "brotli", "zstd", "zip", "parquet", "orc"];
        let auto_compress = match file_path.extension().and_then(|s| s.to_str()) {
            Some(ext) => !compressed_extensions.contains(&ext.to_lowercase().as_str()),
            None => true,
        };

        // Execute Snow CLI
        let mut cmd = std::process::Command::new(snow_cli_path);
        let auto_compress_val = if auto_compress { "TRUE" } else { "FALSE" };
        let put_query = format!(
            "PUT 'file://{}' @\"{}\".\"{}\".\"{}\" AUTO_COMPRESS = {} OVERWRITE = TRUE",
            abs_path_str, self.database, self.schema, stage_name, auto_compress_val
        );

        cmd.arg("sql").arg("-q").arg(put_query.as_str());

        // Snow CLI should pick up other connection details from env vars
        if let Some(pwd) = &self.password {
            cmd.env("SNOWFLAKE_PASSWORD", pwd);
        }

        let output = cmd
            .output()
            .map_err(|e| format!("Failed to execute Snow CLI: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(format!(
                "Snow CLI upload failed.\nStdout: {}\nStderr: {}",
                stdout, stderr
            ));
        }

        let file_name = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file");
        Ok(format!(
            "Successfully uploaded {} to {}",
            file_name, qualified_stage_name
        ))
    }

    pub async fn ai_parse_document(
        &self,
        stage_file_path: &str,
        mode: Option<&str>,
    ) -> Result<String, String> {
        // Parse mode: LAYOUT preserves formatting, OCR for text extraction
        let parse_mode = mode.unwrap_or("LAYOUT");

        // Parse the stage path to build the TO_FILE call correctly
        let query = if stage_file_path.starts_with("@") {
            // Stage path format: @stage_name/file.pdf or @db.schema.stage/file.pdf
            let path = stage_file_path.trim_start_matches('@');

            // Split into stage and file parts
            if let Some(slash_pos) = path.find('/') {
                let stage = &path[..slash_pos];
                let file = &path[slash_pos + 1..];

                // Build the proper AI_PARSE_DOCUMENT query
                format!(
                    "SELECT AI_PARSE_DOCUMENT(TO_FILE('@{}', '{}'), {{'mode': '{}', 'page_split': true}}) as parsed_content",
                    stage, file, parse_mode
                )
            } else {
                return Err(format!(
                    "Invalid stage path format - must include file: {}",
                    stage_file_path
                ));
            }
        } else {
            return Err(format!("Stage path must start with @: {}", stage_file_path));
        };

        if std::env::var("SNOWFLAKE_DEBUG").is_ok() {
            println!("[DEBUG] AI_PARSE_DOCUMENT query: {}", query);
        }

        let results = self
            .session
            .query(query.as_str())
            .await
            .map_err(|e| format!("Failed to parse document: {}", e))?;

        if let Some(row) = results.into_iter().next() {
            // Get the parsed content - it returns a VARIANT/JSON
            let parsed_content: serde_json::Value = row
                .get("parsed_content")
                .map_err(|e| format!("Failed to get parsed content: {}", e))?;

            // Convert to JSON string for Python
            Ok(serde_json::to_string(&parsed_content).map_err(|e| e.to_string())?)
        } else {
            Err("No content returned from AI_PARSE_DOCUMENT".to_string())
        }
    }

    pub async fn create_table_from_file(
        &self,
        stage_path: &str,
        table_name: &str,
        file_type: &str,
    ) -> Result<(), String> {
        let sanitized_table_name = table_name
            .to_uppercase()
            .replace(|c: char| !c.is_alphanumeric() && c != '_', "_");
        let file_type_upper = file_type.to_uppercase();

        if file_type_upper != "JSON" && file_type_upper != "XML" {
            return Err("Invalid file_type specified. Must be 'JSON' or 'XML'.".to_string());
        }

        let file_format_name = format!("FORMAT_{}", sanitized_table_name);
        let create_format_query = if file_type_upper == "XML" {
            format!(
                "CREATE OR REPLACE FILE FORMAT {} TYPE = XML STRIP_OUTER_ELEMENT = TRUE",
                file_format_name
            )
        } else {
            format!(
                "CREATE OR REPLACE FILE FORMAT {} TYPE = JSON STRIP_OUTER_ARRAY = TRUE",
                file_format_name
            )
        };
        self.session
            .query(&*create_format_query)
            .await
            .map_err(|e| e.to_string())?;

        let create_table_query = format!(
            "CREATE OR REPLACE TABLE {} (RAW_DATA VARIANT)",
            sanitized_table_name
        );
        self.session
            .query(&*create_table_query)
            .await
            .map_err(|e| e.to_string())?;

        let copy_into_query = format!(
            "COPY INTO {} FROM '{}' FILE_FORMAT = (FORMAT_NAME = '{}') ON_ERROR = 'CONTINUE'",
            sanitized_table_name, stage_path, file_format_name
        );
        self.session
            .query(copy_into_query.as_str())
            .await
            .map_err(|e| e.to_string())?;

        Ok(())
    }

    pub async fn query_json_table(
        &self,
        table_name: &str,
        json_path: &str,
        variant_column: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        let column = variant_column.unwrap_or("RAW_DATA");
        let sanitized_path = json_path.replace("'", "''");
        let query = format!("SELECT {}:{} FROM {}", column, sanitized_path, table_name);
        self.execute_query(&query, None, None, None).await
    }

    pub async fn query_xml_table(
        &self,
        table_name: &str,
        element_name: &str,
        element_index: Option<i64>,
        variant_column: Option<&str>,
    ) -> Result<serde_json::Value, String> {
        let column = variant_column.unwrap_or("RAW_DATA");

        let select_expression = if let Some(index) = element_index {
            format!("XMLGET({}, '{}', {}):'$'", column, element_name, index)
        } else {
            format!("XMLGET({}, '{}'):'$'", column, element_name)
        };

        // Casting to VARCHAR here to get the content as a string.
        // If the element contains complex XML/JSON, it will be returned as such.
        let query = format!(
            "SELECT {sel_expr}::VARCHAR FROM {table_name}",
            sel_expr = select_expression,
            table_name = table_name
        );

        self.execute_query(&query, None, None, None).await
    }

    pub async fn get_or_create_conversation(
        &self,
        conversation_id: &str,
        settings: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        // Use the actual user for schema name, not conversation_id
        self.setup_user_resources(&self.user).await?;
        let schema_name = _get_user_schema_name(&self.user);
        let table_name = format!("OPENBB_AGENTS.{}.AGENTS_CONVERSATIONS", schema_name);

        let query = format!(
            "SELECT METADATA FROM {} WHERE CONVERSATION_ID = '{}'",
            table_name, conversation_id
        );
        let result = self.execute_statement(&query).await?;

        if let Some(rows) = result.as_array() {
            if !rows.is_empty() {
                if let Some(metadata_str) = rows[0].get("METADATA").and_then(|v| v.as_str()) {
                    if !metadata_str.is_empty() {
                        return serde_json::from_str(metadata_str).map_err(|e| e.to_string());
                    }
                }
            }
        }

        let settings_str = serde_json::to_string(&settings).map_err(|e| e.to_string())?;
        let insert_query = format!(
            "INSERT INTO {} (CONVERSATION_ID, METADATA) SELECT '{}', PARSE_JSON('{}')",
            table_name,
            conversation_id,
            settings_str.replace("'", "''")
        );
        self.execute_statement(&insert_query).await?;
        Ok(settings)
    }

    pub async fn update_conversation_settings(
        &self,
        conversation_id: &str,
        settings: serde_json::Value,
    ) -> Result<(), String> {
        // Use the actual user for schema name
        let schema_name = _get_user_schema_name(&self.user);
        let table_name = format!("OPENBB_AGENTS.{}.AGENTS_CONVERSATIONS", schema_name);
        let settings_str = serde_json::to_string(&settings).map_err(|e| e.to_string())?;
        let query = format!(
            "UPDATE {} SET METADATA = PARSE_JSON('{}'), LAST_UPDATED_AT = CURRENT_TIMESTAMP() WHERE CONVERSATION_ID = '{}'",
            table_name,
            settings_str.replace("'", "''"),
            conversation_id
        );
        self.execute_statement(&query).await?;
        Ok(())
    }

    // Add public getter methods for private fields
    pub async fn get_database(&self) -> Result<String, String> {
        Ok(self.database.clone())
    }

    pub async fn get_schema(&self) -> Result<String, String> {
        Ok(self.schema.clone())
    }

    pub async fn get_user(&self) -> Result<String, String> {
        Ok(self.user.clone())
    }
}
