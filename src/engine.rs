use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt;
use std::path::PathBuf;
use tokio::task;

use snowflake_connector_rs::{SnowflakeClient, SnowflakeClientConfig, SnowflakeAuthMethod, SnowflakeSession, Error};

use crate::agents::AgentsClient;
use crate::cache::CacheConnection;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::{Mutex, OnceCell};

pub static CACHE_DB: OnceCell<Arc<Mutex<CacheConnection>>> = OnceCell::const_new();
static CACHE_INIT_LOCK: Mutex<()> = Mutex::const_new(());
static UPDATING_ALL_TABLES: AtomicBool = AtomicBool::new(false);


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
    password: Option<String>,
    database: String,
    schema: String,
    cortex_enabled: bool,
    semantic_model: Option<SemanticModel>,
    semantic_model_stage: Option<String>,
    semantic_model_file: Option<String>,
    conversation_history: Vec<CortexMessage>,
    jwt_token: Option<String>,
    pub cache: Arc<Mutex<CacheConnection>>,
}

impl fmt::Debug for SnowflakeEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SnowflakeEngine")
            .field("account", &self.account)
            .field("database", &self.database)
            .field("schema", &self.schema)
            .field("cortex_enabled", &self.cortex_enabled)
            .finish()
    }
}

impl SnowflakeEngine {
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

        let cache = if let Some(cache) = CACHE_DB.get() {
            cache.clone()
        } else {
            let _lock = CACHE_INIT_LOCK.lock().await;
            if let Some(cache) = CACHE_DB.get() {
                cache.clone()
            } else {
                let cache_path = std::env::var("SNOWFLAKE_CACHE").ok();
                let cache_key = std::env::var("SNOWFLAKE_CACHE_KEY")
                    .map_err(|_| "SNOWFLAKE_CACHE_KEY environment variable must be set for cache encryption. Please generate a secure, random key (e.g., using `openssl rand -base64 32`) and set it as an environment variable.".to_string())?;
                if cache_key.is_empty() {
                    return Err("SNOWFLAKE_CACHE_KEY environment variable cannot be empty for cache encryption. Please generate a secure, random key (e.g., using `openssl rand -base64 32`) and set it as an environment variable.".to_string());
                }
                let new_cache = Arc::new(Mutex::new(
                    CacheConnection::new(cache_path.as_deref(), &cache_key)
                        .await
                        .map_err(|e| format!("Failed to initialize cache: {}", e))?,
                ));
                CACHE_DB
                    .set(new_cache)
                    .map_err(|_| "Cache already initialized".to_string())?;
                CACHE_DB.get().unwrap().clone()
            }
        };

        Ok(SnowflakeEngine {
            session,
            account: account.to_string(),
            password: Some(password.to_string()),
            database: database.to_string(),
            schema: schema.to_string(),
            cortex_enabled: true,
            semantic_model: None,
            semantic_model_stage: None,
            semantic_model_file: None,
            conversation_history: Vec::new(),
            jwt_token: None,
            cache,
        })
    }


    pub fn generate_jwt(&mut self) -> Result<(), String> {
        // For Cortex API, we need to use Snowflake session token, not the password
        // The password is used for initial authentication, but Cortex needs the session token
        
        // Get the session token from the active session
        let token = self.session.token()
            .ok_or("Failed to get session token. Session may not be authenticated.")?;
        
        self.jwt_token = Some(token.to_string());
        Ok(())
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
        self.session.query(create_stage.as_str()).await.map_err(|e| e.to_string())?;

        let model_content = std::fs::read_to_string(model_path)
            .map_err(|e| format!("Failed to read semantic model: {}", e))?;

        let semantic_model: SemanticModel = serde_yaml::from_str(&model_content)
            .or_else(|_| serde_json::from_str(&model_content))
            .map_err(|e| format!("Invalid semantic model format: {}", e))?;

        // Get the filename from path
        let filename = model_path.file_name()
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
        self.session.query(put_query.as_str()).await.map_err(|e| e.to_string())?;

        self.semantic_model = Some(semantic_model);
        self.semantic_model_stage = Some(format!("@\"{}\".\"{}\".\"{}\"/{}", self.database, self.schema, stage_name, filename));

        Ok(format!("Semantic model uploaded to {}", self.semantic_model_stage.as_ref().unwrap()))
    }

    pub async fn verify_semantic_model(&self) -> Result<Vec<String>, String> {
        let model = self.semantic_model.as_ref().ok_or("No semantic model loaded")?;
        let mut warnings = Vec::new();

        for table in &model.tables {
            let check_query = format!(
                "SELECT COUNT(*) as cnt FROM \"{}\".INFORMATION_SCHEMA.TABLES \n                 WHERE TABLE_SCHEMA = '{}' AND TABLE_NAME = '{}'",
                self.database, self.schema, table.base_table
            );
            let results = self.session.query(check_query.as_str()).await.map_err(|e| e.to_string())?;
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
        request_body_map.insert("semantic_model_file".to_string(), json!(self.semantic_model_stage.as_ref().unwrap()));

        let request_body = serde_json::Value::Object(request_body_map);

        if std::env::var("SNOWFLAKE_DEBUG").is_ok() {
            println!("[CORTEX ANALYST REQUEST BODY]:\n{}", serde_json::to_string_pretty(&request_body).unwrap());
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
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            let request_body_json = serde_json::to_string_pretty(&request_body).unwrap_or_else(|e| format!("Failed to serialize request body for error: {}", e));

            if std::env::var("SNOWFLAKE_DEBUG").is_ok() {
                return Err(format!("Cortex API error {}: {}\nRequest Body: {}", status, error_text, request_body_json));
            } else {
                return Err(format!("Cortex API error {}: {}", status, error_text));
            }
        }

        let response_text = response.text().await
            .map_err(|e| format!("Failed to read response: {}", e))?;

        let cortex_response: CortexAnalystResponse = serde_json::from_str(&response_text)
            .map_err(|e| format!("Failed to parse Cortex response: {}. Body: {}", e, response_text))?;

        if use_history {
            self.conversation_history.push(CortexMessage {
                role: "user".to_string(),
                content: vec![CortexContent::Text { text: message.to_string() }],
            });

            self.conversation_history.push(CortexMessage {
                role: cortex_response.message.role.clone(),
                content: cortex_response.message.content.iter().map(|c| {
                    match c {
                        CortexResponseContent::Text { text } => CortexContent::Text { text: text.clone() },
                        CortexResponseContent::Sql { statement } => CortexContent::Text { text: statement.clone() },
                        CortexResponseContent::Suggestions { suggestions } => {
                            CortexContent::Text { text: suggestions.join(", ") }
                        }
                    }
                }).collect(),
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
                    let schema_json: serde_json::Value = match serde_json::from_str(&schema_json_str) {
                        Ok(json) => json,
                        Err(_) => continue, // Skip if JSON is invalid
                    };

                    if let Some(columns_val) = schema_json.get("columns") {
                        if let Some(columns_arr) = columns_val.as_array() {
                            if !columns_arr.is_empty() {
                                let full_table_name = format!(
                                    "{}.{}.{}",
                                    self.database, self.schema, table_name
                                );
                                let column_names: Vec<String> = columns_arr.iter()
                                    .filter_map(|c| c.get("COLUMN_NAME").and_then(|n| n.as_str()).map(String::from))
                                    .collect();
                                context.push_str(&format!("-- {}({})\n", full_table_name, column_names.join(", ")));
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

        let stage_name = std::env::var("SNOWFLAKE_STAGE")
            .unwrap_or_else(|_| "SEMANTIC_MODELS".to_string());

        // Construct the stage path: @DATABASE.SCHEMA.STAGE/filename
        let stage_path = format!("@\"{}\".\"{}\".\"{}\"/{}", self.database, self.schema, stage_name, file_name);

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
        let schema = self.get_schema().await?;
        let mut tables_map: std::collections::HashMap<String, Vec<Schema>> = std::collections::HashMap::new();

        for col in schema {
            tables_map.entry(col.table_name.clone()).or_default().push(col);
        }

        let mut semantic_tables = Vec::new();
        for (table_name, columns) in tables_map {
            let dimensions: Vec<SemanticColumn> = columns.iter().map(|col| {
                SemanticColumn {
                    name: col.column_name.clone(),
                    description: Some(format!("{} column from {}", col.column_name, table_name)),
                    data_type: col.data_type.clone(),
                    expr: None,
                    synonyms: None,
                    sample_values: None,
                }
            }).collect();

            semantic_tables.push(SemanticTable {
                name: table_name.clone(),
                description: Some(format!("Auto-generated semantic model for {}", table_name)),
                base_table: format!("\"{}\".\"{}\".\"{}\"", self.database, self.schema, table_name),
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
        self.session.query(explain_query.as_str()).await.map_err(|e| e.to_string())?;
        Ok(())
    }

    pub async fn get_query_suggestions(&mut self, partial_query: &str) -> Result<Vec<String>, String> {
        if !self.is_cortex_enabled() {
            return Ok(vec![]);
        }

        let mut full_prompt = partial_query.to_string();
        if let Ok(context) = self.get_database_context_string().await {
            if !context.is_empty() {
                full_prompt = format!("{}\n\nComplete this SQL query, ensuring all table names are fully qualified (e.g., DATABASE.SCHEMA.TABLE): {}", context, partial_query);
            }
        }

        match self.chat_with_analyst(
            &full_prompt,
            false // Don't use history for suggestions
        ).await {
            Ok(response) => {
                for content in response.message.content {
                    match content {
                        CortexResponseContent::Suggestions { suggestions } => return Ok(suggestions),
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
            Err(_) => Ok(vec![]) // Silently fail for suggestions
        }
    }

    fn build_where_clause(filter_model: &serde_json::Value) -> String {
        let mut where_clauses = Vec::new();
        if let Some(map) = filter_model.as_object() {
            for (column, filter) in map {
                if let Some(filter_type) = filter.get("filterType").and_then(|v| v.as_str()) {
                    match filter_type {
                        "text" => {
                            if let Some(filter_value) = filter.get("filter").and_then(|v| v.as_str()) {
                                where_clauses.push(format!("{} LIKE '%{}%'", column, filter_value));
                            }
                        }
                        "number" => {
                            if let Some(filter_value) = filter.get("filter").and_then(|v| v.as_i64()) {
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

        let where_clause = filter_model.map_or(String::new(), |fm| Self::build_where_clause(fm));
        let filtered_query = if where_clause.is_empty() {
            cleaned_query
        } else {
            format!("{} {}", cleaned_query, where_clause)
        };

        let count_query = format!("SELECT COUNT(*) as count FROM ({})", filtered_query);
        let count_results = self.session.query(count_query.as_str()).await.map_err(|e| e.to_string())?;
        let row_count: i64 = count_results[0].get("count").unwrap_or(0);

        let paginated_query = if let (Some(start), Some(end)) = (start_row, end_row) {
            format!(
                "{} LIMIT {} OFFSET {}",
                filtered_query,
                end - start,
                start
            )
        } else {
            filtered_query
        };
        let results = self.session.query(paginated_query.as_str()).await.map_err(|e| e.to_string())?;

        let column_defs: Vec<_> = results.iter().next().map_or(vec![], |first_row| {
            first_row.column_types().into_iter().map(|col| {
                json!({
                    "field": col.name(),
                    "type": col.column_type().snowflake_type(),
                })
            }).collect()
        });

        let column_names: Vec<String> = column_defs.iter()
            .map(|def| def["field"].as_str().unwrap_or_default().to_string())
            .collect();

        let row_data: Vec<_> = results.into_iter().map(|row| {
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
        }).collect();

        Ok(json!({
            "columnDefs": column_defs,
            "rowData": row_data,
            "rowCount": if start_row.is_some() && end_row.is_some() { 1 } else { row_count },
        }))
    }

    pub async fn get_schema(&self) -> Result<Vec<Schema>, String> {
        let query = format!(
            "SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, \n             ORDINAL_POSITION, COLUMN_DEFAULT, IS_NULLABLE, DATA_TYPE \n             FROM \"{}\".INFORMATION_SCHEMA.COLUMNS \n             WHERE TABLE_SCHEMA = '{}' \n             ORDER BY TABLE_NAME, ORDINAL_POSITION",
            self.database, self.schema
        );

        let results = self.session.query(query.as_str()).await.map_err(|e| e.to_string())?;
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
        // Check cache first
        let cache_lock = self.cache.lock().await;
        if let Ok(Some(cached_columns)) = cache_lock.get_table_schema(catalog, schema, table).await {
            drop(cache_lock);

            // Convert ColumnInfo back to JSON format
            let columns_json: Vec<serde_json::Value> = cached_columns.into_iter().map(|col| {
                json!({
                    "COLUMN_NAME": col.column_name,
                    "DATA_TYPE": col.data_type,
                    "IS_NULLABLE": col.is_nullable,
                    "COLUMN_DEFAULT": col.column_default,
                    "ORDINAL_POSITION": col.ordinal_position,
                })
            }).collect();

            return Ok(serde_json::to_string(&json!({
                "columns": columns_json,
                "primary_keys": [],
                "foreign_keys": [],
            })).unwrap());
        }
        drop(cache_lock);

        // Cache miss - fetch from Snowflake
        let schema_json = self.get_table_info_direct(table_name).await?;

        // Extract columns and cache them
        if let Some(columns_arr) = schema_json.get("columns").and_then(|c| c.as_array()) {
            let column_infos: Vec<crate::cache::ColumnInfo> = columns_arr.iter()
                .enumerate()
                .filter_map(|(idx, col)| {
                    Some(crate::cache::ColumnInfo {
                        column_name: col.get("COLUMN_NAME")?.as_str()?.to_string(),
                        ordinal_position: idx as i64,
                        column_default: col.get("COLUMN_DEFAULT").and_then(|v| v.as_str()).map(String::from),
                        is_nullable: col.get("IS_NULLABLE")?.as_str()?.to_string(),
                        data_type: col.get("DATA_TYPE")?.as_str()?.to_string(),
                    })
                })
                .collect();

            if !column_infos.is_empty() {
                let cache = self.cache.clone();
                let cat = catalog.to_string();
                let sch = schema.to_string();
                let tbl = table.to_string();
                task::spawn(async move {
                    let cache_lock = cache.lock().await;
                    let _ = cache_lock.set_table_schema(cat, sch, tbl, column_infos).await;
                });
            }
        }

        serde_json::to_string(&schema_json).map_err(|e| e.to_string())
    }

    async fn get_table_info_direct(&self, table_name: &str) -> Result<serde_json::Value, String> {
        let parts: Vec<&str> = table_name.split('.').collect();
        let (catalog, schema, table) = match parts.as_slice() {
            [c, s, t] => (*c, *s, *t),
            [s, t] => (self.database.as_str(), *s, *t),
            [t] => (self.database.as_str(), self.schema.as_str(), *t),
            _ => return Err("Invalid table name format".to_string()),
        };

        let full_table_name = format!("\"{}\".\"{}\".\"{}\"", catalog, schema, table);
        let describe_query = format!("DESCRIBE TABLE {}", full_table_name);

        let describe_results = self.session.query(describe_query.as_str()).await
            .map_err(|e| format!("Failed to describe table: {}", e))?;

        // Convert each row to JSON with ALL metadata columns
        let columns_json: Vec<serde_json::Value> = describe_results.into_iter()
            .filter_map(|row| {
                let mut col = serde_json::Map::new();
                col.insert("name".to_string(), json!(row.get::<String>("name").ok()?));
                col.insert("type".to_string(), json!(row.get::<String>("type").ok()?));
                col.insert("kind".to_string(), json!(row.get::<String>("kind").unwrap_or_default()));
                col.insert("null?".to_string(), json!(row.get::<String>("null?").unwrap_or_default()));
                col.insert("default".to_string(), json!(row.get::<Option<String>>("default").ok().flatten()));
                col.insert("primary key".to_string(), json!(row.get::<String>("primary key").unwrap_or_default()));
                col.insert("unique key".to_string(), json!(row.get::<String>("unique key").unwrap_or_default()));
                col.insert("check".to_string(), json!(row.get::<Option<String>>("check").ok().flatten()));
                col.insert("expression".to_string(), json!(row.get::<Option<String>>("expression").ok().flatten()));
                col.insert("comment".to_string(), json!(row.get::<Option<String>>("comment").ok().flatten()));
                col.insert("policy name".to_string(), json!(row.get::<Option<String>>("policy name").ok().flatten()));
                col.insert("privacy domain".to_string(), json!(row.get::<Option<String>>("privacy domain").ok().flatten()));
                Some(serde_json::Value::Object(col))
            })
            .collect();

        Ok(serde_json::Value::Array(columns_json))
    }

    /// Get list of tables in current schema
    pub async fn get_table_list(&self) -> Result<Vec<String>, String> {
        let cache_lock = self.cache.lock().await;
        if let Ok(tables) = cache_lock.get_tables_in_schema(&self.database, &self.schema).await {
            if !tables.is_empty() {
                return Ok(tables);
            }
        }
        drop(cache_lock); // Release the lock before potentially awaiting another async call

        let tables = self.list_tables_in_current_schema_direct().await.map_err(|e| e.to_string())?;
        let cache = self.cache.clone();
        let db = self.database.clone();
        let s = self.schema.clone();
        let tables_clone = tables.clone();
        task::spawn(async move {
            let cache_lock = cache.lock().await;
            let _ = cache_lock.set_tables_in_schema(db, s, tables_clone).await;
        });
        Ok(tables)
    }

    pub fn is_cortex_enabled(&self) -> bool {
        self.cortex_enabled && self.password.is_some()
    }

    pub fn get_semantic_model(&self) -> Option<&SemanticModel> {
        self.semantic_model.as_ref()
    }

    pub fn has_semantic_model_stage(&self) -> bool {
        self.semantic_model_stage.is_some()
    }

    /// List all stages available to the user
    pub async fn list_stages(&self) -> Result<Vec<String>, String> {
        let cache_lock = self.cache.lock().await;
        if let Ok(stages) = cache_lock.get_stages().await {
            if !stages.is_empty() {
                return Ok(stages);
            }
        }
        drop(cache_lock); // Release the lock before potentially awaiting another async call

        let stages = self.list_stages_direct().await.map_err(|e| e.to_string())?;
        let cache = self.cache.clone();
        let stages_clone = stages.clone();
        task::spawn(async move {
            let cache_lock = cache.lock().await;
            let _ = cache_lock.set_stages(stages_clone).await;
        });
        Ok(stages)
    }

    /// List files in a specific stage
    pub async fn list_files_in_stage(&self, stage_name: &str) -> Result<Vec<String>, String> {
        let cache_lock = self.cache.lock().await;
        if let Ok(files) = cache_lock.get_stage_files(stage_name).await {
            if !files.is_empty() {
                return Ok(files);
            }
        }
        drop(cache_lock); // Release the lock before potentially awaiting another async call

        let files = self.list_files_in_stage_direct(stage_name).await.map_err(|e| e.to_string())?;
        let cache = self.cache.clone();
        let stage_name_clone = stage_name.to_string();
        let files_clone = files.clone();
        task::spawn(async move {
            let cache_lock = cache.lock().await;
            let _ = cache_lock.set_stage_files(stage_name_clone, files_clone).await;
        });
        Ok(files)
    }

    /// Switch to a different semantic model stage and file
    pub fn switch_semantic_model(&mut self, stage: String, file: String) {
        let stage_path = format!("@\"{}\".\"{}\".\"{}\"/{}", self.database, self.schema, stage, file);
        self.semantic_model_stage = Some(stage_path);
        self.semantic_model_file = Some(file);
    }

    /// Get current semantic model stage and file
    pub fn get_semantic_model_config(&self) -> (Option<&str>, Option<&str>) {
        (
            self.semantic_model_stage.as_deref(),
            self.semantic_model_file.as_deref()
        )
    }

    /// Execute AI_COMPLETE function with database context
    pub async fn ai_complete(
        &self,
        model: &str,
        prompt: &str,
        options: Option<serde_json::Value>,
        tools: Option<Vec<crate::agents::Tool>>,
    ) -> Result<String, String> {
        let mut full_prompt = prompt.to_string();
        if let Ok(context) = self.get_database_context_string().await {
            if !context.is_empty() {
                full_prompt = format!("{}\n\n{}", context, prompt);
            }
        }
        self.ai_complete_raw(model, &full_prompt, options, tools).await
    }

    /// Execute AI_COMPLETE function without database context (faster)
    pub async fn ai_complete_raw(
        &self,
        model: &str,
        prompt: &str,
        options: Option<serde_json::Value>,
        tools: Option<Vec<crate::agents::Tool>>,
    ) -> Result<String, String> {
        let tools_json = if let Some(t) = tools {
            serde_json::to_string(&t).map_err(|e| e.to_string())?
        } else {
            "NULL".to_string()
        };

        let query = if let Some(opts) = options {
            let mut option_pairs = Vec::new();
            if let Some(obj) = opts.as_object() {
                for (key, value) in obj {
                    let val_str = match value {
                        serde_json::Value::Number(n) => n.to_string(),
                        serde_json::Value::String(s) => format!("'{}'", s.replace("'", "''")),
                        serde_json::Value::Bool(b) => b.to_string(),
                        _ => value.to_string(),
                    };
                    option_pairs.push(format!("'{}', {}", key, val_str));
                }
            }
            let options_str = option_pairs.join(", ");

            format!(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE('{}', '{}', OBJECT_CONSTRUCT({}, 'tools', PARSE_JSON('{}'))) AS response",
                model,
                prompt.replace("'", "''"),
                options_str,
                tools_json
            )
        } else {
            format!(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE('{}', '{}', OBJECT_CONSTRUCT('tools', PARSE_JSON('{}'))) AS response",
                model,
                prompt.replace("'", "''"),
                tools_json
            )
        };

        let results = self.session.query(query.as_str()).await.map_err(|e| e.to_string())?;

        if let Some(row) = results.into_iter().next() {
            row.get::<String>("response").map_err(|e| e.to_string())
        } else {
            Err("No response from CORTEX.COMPLETE".to_string())
        }
    }

    /// Get list of available AI models from Snowflake
    pub async fn get_available_models(&self) -> Result<Vec<(String, String)>, String> {
        // Query Snowflake for available models
        let query = "SELECT * FROM TABLE(CORTEX.COMPLETE.SHOW_AVAILABLE_MODELS())";

        match self.session.query(query).await {
            Ok(results) => {
                let mut models = Vec::new();
                for row in results {
                    if let (Ok(name), Ok(desc)) = (
                        row.get::<String>("MODEL_NAME"),
                        row.get::<String>("DESCRIPTION")
                    ) {
                        models.push((name, desc));
                    }
                }

                // If query succeeds but returns empty, fallback to documented models
                if models.is_empty() {
                    Ok(self.get_documented_models())
                } else {
                    Ok(models)
                }
            }
            Err(_) => {
                // If query fails (maybe no permissions), return documented models
                Ok(self.get_documented_models())
            }
        }
    }

    fn get_documented_models(&self) -> Vec<(String, String)> {
        // REAL models from https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api#model-availability
        vec![
            ("claude-4-sonnet".to_string(), "Claude 4 Sonnet".to_string()),
            ("claude-4-opus".to_string(), "Claude 4 Opus".to_string()),
            ("claude-3-7-sonnet".to_string(), "Claude 3.7 Sonnet".to_string()),
            ("claude-3-5-sonnet".to_string(), "Claude 3.5 Sonnet".to_string()),
            ("openai-gpt-4.1".to_string(), "OpenAI GPT-4.1".to_string()),
            ("openai-o4-mini".to_string(), "OpenAI o4-mini".to_string()),
            ("openai-gpt-5-chat".to_string(), "OpenAI GPT-5 Chat".to_string()),
            ("llama4-maverick".to_string(), "Llama 4 Maverick".to_string()),
            ("llama3.1-8b".to_string(), "Llama 3.1 8B".to_string()),
            ("llama3.1-70b".to_string(), "Llama 3.1 70B".to_string()),
            ("llama3.1-405b".to_string(), "Llama 3.1 405B".to_string()),
            ("deepseek-r1".to_string(), "DeepSeek R1".to_string()),
            ("mistral-7b".to_string(), "Mistral 7B".to_string()),
            ("mistral-large".to_string(), "Mistral Large".to_string()),
            ("mistral-large2".to_string(), "Mistral Large 2".to_string()),
            ("snowflake-llama-3.3-70b".to_string(), "Snowflake Llama 3.3 70B".to_string()),
        ]
    }

    /// Create an AgentsClient for streaming complete interactions
    pub fn create_agents_client(&mut self) -> Result<AgentsClient, String> {
        // Generate/set token if we don't have one
        if self.jwt_token.is_none() {
            self.generate_jwt()?;
        }
        let token = self.jwt_token.as_ref().unwrap().clone();

        Ok(AgentsClient::new(
            self.account.clone(),
            token,
            self.database.clone(),
            self.schema.clone(),
        ))
    }

    /// Get current database
    pub fn get_current_database(&self) -> &str {
        &self.database
    }

    /// Get current schema
    pub fn get_current_schema(&self) -> &str {
        &self.schema
    }

    /// Get current account
    pub fn get_current_account(&self) -> &str {
        &self.account
    }

    /// List all databases accessible to the user
    pub async fn list_databases(&self) -> Result<Vec<String>, String> {
        let cache_lock = self.cache.lock().await;
        if let Ok(databases) = cache_lock.get_databases().await {
            if !databases.is_empty() {
                return Ok(databases);
            }
        }
        drop(cache_lock); // Release the lock before potentially awaiting another async call

        let databases = self.list_databases_direct().await.map_err(|e| e.to_string())?;
        let cache = self.cache.clone();
        let databases_clone = databases.clone();
        task::spawn(async move {
            let cache_lock = cache.lock().await;
            let _ = cache_lock.set_databases(databases_clone).await;
        });
        Ok(databases)
    }

    /// List all schemas in a specific database (or current if None)
    pub async fn list_schemas(&self, database: Option<&str>) -> Result<Vec<String>, String> {
        let db_to_filter = database.unwrap_or(&self.database);

        let cache_lock = self.cache.lock().await;
        if let Ok(schemas) = cache_lock.get_schemas(db_to_filter).await {
            if !schemas.is_empty() {
                return Ok(schemas);
            }
        }
        drop(cache_lock); // Release the lock before potentially awaiting another async call

        let schemas = self.list_schemas_direct(Some(db_to_filter)).await.map_err(|e| e.to_string())?;
        let cache = self.cache.clone();
        let db_to_filter_clone = db_to_filter.to_string();
        let schemas_clone = schemas.clone();
        task::spawn(async move {
            let cache_lock = cache.lock().await;
            let _ = cache_lock.set_schemas(db_to_filter_clone, schemas_clone).await;
        });
        Ok(schemas)
    }

    /// List all warehouses accessible to the user
    pub async fn list_warehouses(&self) -> Result<Vec<String>, String> {
        let cache_lock = self.cache.lock().await;
        if let Ok(warehouses) = cache_lock.get_warehouses().await {
            if !warehouses.is_empty() {
                return Ok(warehouses);
            }
        }
        drop(cache_lock); // Release the lock before potentially awaiting another async call

        let warehouses = self.list_warehouses_direct().await.map_err(|e| e.to_string())?;
        let cache = self.cache.clone();
        let warehouses_clone = warehouses.clone();
        task::spawn(async move {
            let cache_lock = cache.lock().await;
            let _ = cache_lock.set_warehouses(warehouses_clone).await;
        });
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
        let cache_lock = self.cache.lock().await;
        if let Ok(tables) = cache_lock.get_all_tables().await {
            if !tables.is_empty() {
                return Ok(tables);
            }
        }
        drop(cache_lock); // Release the lock before potentially awaiting another async call

        // If cache is empty, check if an update is already in progress
        if UPDATING_ALL_TABLES.load(Ordering::SeqCst) {
            return Ok(Vec::new()); // Return empty immediately if an update is in progress
        }

        // If no update in progress, set the flag and spawn a background task
        UPDATING_ALL_TABLES.store(true, Ordering::SeqCst);
        let cache = self.cache.clone();
        let session = self.session.clone();
        let database = self.database.clone();
        let schema = self.schema.clone();

        task::spawn(async move {
            let engine = SnowflakeEngine {
                session,
                account: String::new(), // Not used in list_all_tables_direct
                password: None, // Not used
                database,
                schema,
                cortex_enabled: false, // Not used
                semantic_model: None, // Not used
                semantic_model_stage: None, // Not used
                semantic_model_file: None, // Not used
                conversation_history: Vec::new(), // Not used
                jwt_token: None, // Not used
                cache: cache.clone(),
            };
            match engine.list_all_tables_direct().await {
                Ok(tables) => {
                    let cache_lock = cache.lock().await;
                    let _ = cache_lock.set_all_tables(tables).await;
                }
                Err(e) => {
                    eprintln!("Error updating all tables in background: {}", e);
                }
            }
            UPDATING_ALL_TABLES.store(false, Ordering::SeqCst);
        });

        Ok(Vec::new()) // Return empty immediately
    }

    /// List tables in a specific database and schema
    pub async fn list_tables_in(&self, database: &str, schema: &str) -> Result<Vec<String>, String> {
        let cache_lock = self.cache.lock().await;
        if let Ok(tables) = cache_lock.get_tables_in_schema(database, schema).await {
            if !tables.is_empty() {
                return Ok(tables);
            }
        }
        drop(cache_lock); // Release the lock before potentially awaiting another async call

        let tables = self.list_tables_in_direct(database, schema).await.map_err(|e| e.to_string())?;
        let cache = self.cache.clone();
        let db = database.to_string();
        let s = schema.to_string();
        let tables_clone = tables.clone();
        task::spawn(async move {
            let cache_lock = cache.lock().await;
            let _ = cache_lock.set_tables_in_schema(db, s, tables_clone).await;
        });
        Ok(tables)
    }

    async fn list_tables_in_current_schema_direct(&self) -> Result<Vec<String>, Error> {
        let query = format!(
            "SELECT TABLE_NAME FROM \"{}\".INFORMATION_SCHEMA.TABLES \
             WHERE TABLE_SCHEMA = '{}' \
             ORDER BY TABLE_NAME",
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

    async fn list_tables_in_direct(&self, database: &str, schema: &str) -> Result<Vec<String>, Error> {
        let query = format!(
            "SELECT TABLE_NAME FROM \"{}\".INFORMATION_SCHEMA.TABLES \
             WHERE TABLE_SCHEMA = '{}' \
             ORDER BY TABLE_NAME",
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
                let filename = name.split('/').last().unwrap_or(&name);
                files.push(filename.to_string());
            }
        }
        Ok(files)
    }

    async fn list_databases_direct(&self) -> Result<Vec<String>, Error> {
        let query = "SHOW DATABASES";
        let results = self.session.query(query).await?;

        let mut databases = Vec::new();
        for row in results {
            if let Ok(name) = row.get::<String>("name") {
                databases.push(name);
            }
        }
        Ok(databases)
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

    async fn list_all_tables_direct(&self) -> Result<Vec<(String, String, String)>, Error> {
        let databases = self.list_databases_direct().await?;
        let mut all_tables = Vec::new();

        for db in databases {
            match self.list_schemas_direct(Some(&db)).await {
                Ok(schemas) => {
                    for schema in &schemas {
                        let query = format!(
                            "SELECT TABLE_CATALOG, TABLE_SCHEMA, TABLE_NAME \
                             FROM \"{}\".INFORMATION_SCHEMA.TABLES \
                             WHERE TABLE_SCHEMA = '{}' \
                             ORDER BY TABLE_NAME",
                            db, schema
                        );

                        match self.session.query(query.as_str()).await {
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
                            Err(_) => continue, // Skip schemas we can't access
                        }
                    }
                }
                Err(_) => continue, // Skip databases we can't access
            }
        }

        Ok(all_tables)
    }

    pub async fn close(&mut self) -> Result<(), String> {
        let mut cache = self.cache.lock().await;
        cache.close().await.map_err(|e| e.to_string())
    }
}
