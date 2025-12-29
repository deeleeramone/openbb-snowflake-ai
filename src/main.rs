mod agentic;
mod agents;
mod engine;
mod events;
mod jobs;
mod lsp;
mod tools;

use crate::agentic::{extract_sql_snippets, run_agentic_loop};
use crate::engine::SnowflakeEngine;
use crate::events::{create_emitter, EventEmitter};
use crate::jobs::JobManager;
use crate::tools::ToolExecutor;

use clap::{Parser, Subcommand};
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Config, Context, Editor, Helper};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tokio::io::{self, AsyncBufReadExt};
use uuid::Uuid;

// ============================================================================
// Chat Completer for rustyline
// ============================================================================

/// Slash commands available in chat
const SLASH_COMMANDS: &[&str] = &[
    "/exit",
    "/help",
    "/current",
    "/context",
    "/databases",
    "/schemas",
    "/warehouses",
    "/tables",
    "/all_tables",
    "/describe",
    "/complete",
    "/suggest_query",
    "/use_database",
    "/use_warehouse",
    "/use_schema",
    "/use_conversation",
    "/history",
    "/execute",
    "/clear",
    "/reset",
    "/new",
    "/list",
    "/delete",
    "/upload",
    "/documents",
    "/download_document",
    "/jobs",
    "/cancel",
    "/timeout",
    "/tools",
    "/conversations",
    "/label",
    "/model",
    "/models",
    "/temperature",
    "/max_tokens",
    "/stages",
    "/stage_files",
    "/parse",
    "/text2sql",
    "/stats",
];

/// Helper struct for rustyline completions
struct ChatHelper {
    /// Cached list of databases
    databases: Vec<String>,
    /// Cached list of schemas  
    schemas: Vec<String>,
    /// Cached list of tables
    tables: Vec<String>,
    /// Cached list of models
    models: Vec<String>,
    /// Cached list of warehouses
    warehouses: Vec<String>,
    /// Cached list of conversations
    conversations: Vec<String>,
}

impl ChatHelper {
    fn new() -> Self {
        Self {
            databases: Vec::new(),
            schemas: Vec::new(),
            tables: Vec::new(),
            models: Vec::new(),
            warehouses: Vec::new(),
            conversations: Vec::new(),
        }
    }

    fn update_databases(&mut self, databases: Vec<String>) {
        self.databases = databases;
    }

    fn update_schemas(&mut self, schemas: Vec<String>) {
        self.schemas = schemas;
    }

    fn update_tables(&mut self, tables: Vec<String>) {
        self.tables = tables;
    }

    fn update_models(&mut self, models: Vec<String>) {
        self.models = models;
    }

    fn update_warehouses(&mut self, warehouses: Vec<String>) {
        self.warehouses = warehouses;
    }

    fn update_conversations(&mut self, conversations: Vec<String>) {
        self.conversations = conversations;
    }
}

impl Completer for ChatHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let line_to_cursor = &line[..pos];

        // Complete slash commands
        if line_to_cursor.starts_with('/') {
            // /use_database <database> - complete with database names
            if line_to_cursor.starts_with("/use_database ") {
                let prefix = line_to_cursor.strip_prefix("/use_database ").unwrap_or("");
                let start = pos - prefix.len();
                let matches: Vec<Pair> = self
                    .databases
                    .iter()
                    .filter(|db| db.to_uppercase().starts_with(&prefix.to_uppercase()))
                    .map(|db| Pair {
                        display: db.clone(),
                        replacement: db.clone(),
                    })
                    .collect();
                return Ok((start, matches));
            }

            // /schemas <database> - complete with database names (optional db arg)
            if line_to_cursor.starts_with("/schemas ") {
                let prefix = line_to_cursor.strip_prefix("/schemas ").unwrap_or("");
                let start = pos - prefix.len();
                let matches: Vec<Pair> = self
                    .databases
                    .iter()
                    .filter(|db| db.to_uppercase().starts_with(&prefix.to_uppercase()))
                    .map(|db| Pair {
                        display: db.clone(),
                        replacement: db.clone(),
                    })
                    .collect();
                return Ok((start, matches));
            }

            // /use_schema <schema> - complete with schema names
            if line_to_cursor.starts_with("/use_schema ") {
                let prefix = line_to_cursor.strip_prefix("/use_schema ").unwrap_or("");
                let start = pos - prefix.len();
                let matches: Vec<Pair> = self
                    .schemas
                    .iter()
                    .filter(|s| s.to_uppercase().starts_with(&prefix.to_uppercase()))
                    .map(|s| Pair {
                        display: s.clone(),
                        replacement: s.clone(),
                    })
                    .collect();
                return Ok((start, matches));
            }

            // /describe <table> - complete with table names
            if line_to_cursor.starts_with("/describe ") {
                let prefix = line_to_cursor.strip_prefix("/describe ").unwrap_or("");
                let start = pos - prefix.len();
                let matches: Vec<Pair> = self
                    .tables
                    .iter()
                    .filter(|t| t.to_uppercase().starts_with(&prefix.to_uppercase()))
                    .map(|t| Pair {
                        display: t.clone(),
                        replacement: t.clone(),
                    })
                    .collect();
                return Ok((start, matches));
            }

            // /model <model> - complete with model names
            if line_to_cursor.starts_with("/model ") {
                let prefix = line_to_cursor.strip_prefix("/model ").unwrap_or("");
                let start = pos - prefix.len();
                let matches: Vec<Pair> = self
                    .models
                    .iter()
                    .filter(|m| m.to_uppercase().starts_with(&prefix.to_uppercase()))
                    .map(|m| Pair {
                        display: m.clone(),
                        replacement: m.clone(),
                    })
                    .collect();
                return Ok((start, matches));
            }

            // /use_warehouse <warehouse> - complete with warehouse names
            if line_to_cursor.starts_with("/use_warehouse ") {
                let prefix = line_to_cursor.strip_prefix("/use_warehouse ").unwrap_or("");
                let start = pos - prefix.len();
                let matches: Vec<Pair> = self
                    .warehouses
                    .iter()
                    .filter(|w| w.to_uppercase().starts_with(&prefix.to_uppercase()))
                    .map(|w| Pair {
                        display: w.clone(),
                        replacement: w.clone(),
                    })
                    .collect();
                return Ok((start, matches));
            }

            // /use_conversation <conversation> - complete with conversation IDs
            if line_to_cursor.starts_with("/use_conversation ") {
                let prefix = line_to_cursor
                    .strip_prefix("/use_conversation ")
                    .unwrap_or("");
                let start = pos - prefix.len();
                let matches: Vec<Pair> = self
                    .conversations
                    .iter()
                    .filter(|c| c.to_uppercase().starts_with(&prefix.to_uppercase()))
                    .map(|c| Pair {
                        display: c.clone(),
                        replacement: c.clone(),
                    })
                    .collect();
                return Ok((start, matches));
            }

            // Complete the slash command itself
            let matches: Vec<Pair> = SLASH_COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(line_to_cursor))
                .map(|cmd| Pair {
                    display: cmd.to_string(),
                    replacement: cmd.to_string(),
                })
                .collect();
            return Ok((0, matches));
        }

        Ok((pos, Vec::new()))
    }
}

impl Hinter for ChatHelper {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &Context<'_>) -> Option<String> {
        None
    }
}

impl Highlighter for ChatHelper {}
impl Validator for ChatHelper {}
impl Helper for ChatHelper {}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Starts the language server
    Lsp {
        #[arg(short, long, env = "SNOWFLAKE_USER")]
        user: String,

        #[arg(short, long, env = "SNOWFLAKE_PASSWORD")]
        password: String,

        #[arg(short, long, env = "SNOWFLAKE_ACCOUNT")]
        account: String,

        #[arg(short, long, env = "SNOWFLAKE_ROLE")]
        role: String,

        #[arg(short, long, env = "SNOWFLAKE_WAREHOUSE")]
        warehouse: String,

        #[arg(short, long, env = "SNOWFLAKE_DATABASE")]
        database: String,

        #[arg(short, long, env = "SNOWFLAKE_SCHEMA")]
        schema: String,

        #[arg(short, long)]
        env_file: Option<PathBuf>,
    },
    /// Execute SQL queries interactively
    Execute {
        #[arg(short, long, env = "SNOWFLAKE_USER")]
        user: String,

        #[arg(short, long, env = "SNOWFLAKE_PASSWORD")]
        password: String,

        #[arg(short, long, env = "SNOWFLAKE_ACCOUNT")]
        account: String,

        #[arg(short, long, env = "SNOWFLAKE_ROLE")]
        role: String,

        #[arg(short, long, env = "SNOWFLAKE_WAREHOUSE")]
        warehouse: String,

        #[arg(short, long, env = "SNOWFLAKE_DATABASE")]
        database: String,

        #[arg(short, long, env = "SNOWFLAKE_SCHEMA")]
        schema: String,

        #[arg(short, long)]
        env_file: Option<PathBuf>,
    },
    /// Validate SQL queries
    Validate {
        #[arg(short, long, env = "SNOWFLAKE_USER")]
        user: String,

        #[arg(short, long, env = "SNOWFLAKE_PASSWORD")]
        password: String,

        #[arg(short, long, env = "SNOWFLAKE_ACCOUNT")]
        account: String,

        #[arg(short, long, env = "SNOWFLAKE_ROLE")]
        role: String,

        #[arg(short, long, env = "SNOWFLAKE_WAREHOUSE")]
        warehouse: String,

        #[arg(short, long, env = "SNOWFLAKE_DATABASE")]
        database: String,

        #[arg(short, long, env = "SNOWFLAKE_SCHEMA")]
        schema: String,

        #[arg(short, long)]
        env_file: Option<PathBuf>,
    },
    /// Interactive chat with Cortex Analyst (multi-turn conversation)
    Chat {
        #[arg(short, long, env = "SNOWFLAKE_USER")]
        user: String,
        #[arg(short, long, env = "SNOWFLAKE_PASSWORD")]
        password: String,
        #[arg(short, long, env = "SNOWFLAKE_ACCOUNT")]
        account: String,
        #[arg(short, long, env = "SNOWFLAKE_ROLE")]
        role: String,
        #[arg(short, long, env = "SNOWFLAKE_WAREHOUSE")]
        warehouse: String,
        #[arg(short, long, env = "SNOWFLAKE_DATABASE")]
        database: String,
        #[arg(short = 's', long, env = "SNOWFLAKE_SCHEMA")]
        schema: String,
        #[arg(short = 'm', long)]
        semantic_model: Option<PathBuf>,
        #[arg(long)]
        conversation_id: Option<String>,
        #[arg(long)]
        conversation_label: Option<String>,
        #[arg(long)]
        list_conversations: bool,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        temperature: Option<f32>,
        #[arg(long)]
        max_tokens: Option<i32>,
        /// Output events as JSON (machine-parseable)
        #[arg(long)]
        json: bool,
        /// Default timeout in seconds for operations
        #[arg(long, default_value = "120")]
        timeout: u64,
        #[arg(short = 'e', long)]
        env_file: Option<PathBuf>,
    },
}

fn print_chat_help() {
    println!("Commands:");
    println!("  /help                     - Show this command menu");
    println!("  /reset                    - Clear conversation history");
    println!("  /history                  - Show conversation history");
    println!("  /conversations            - List stored conversations");
    println!("  /use_conversation <id>    - Switch to an existing conversation by ID or label");
    println!("  /label [value]            - Set or clear the active conversation label");
    println!("  /model <name>             - Override the active model");
    println!("  /temperature <val>        - Set sampling temperature (0.0-1.0)");
    println!("  /max_tokens <val>         - Set a max token budget");
    println!("  /models                   - List available Cortex models");
    println!("  /execute [sql]            - Run SQL inline or enter multiline prompt");
    println!("  /databases                - List all accessible databases");
    println!("  /schemas [db]             - List schemas in current/specified database");
    println!("  /warehouses               - List all accessible warehouses");
    println!("  /tables                   - List tables in current database/schema");
    println!("  /all_tables               - List all tables across all databases");
    println!("  /describe <table_name>    - Describe a table");
    println!("  /complete <partial_query> - Complete a SQL query using AI");
    println!(
        "  /suggest_query <natural_language_query> - Suggest a SQL query from natural language"
    );
    println!("  /upload <file_path>       - Upload a local file to CORTEX_UPLOADS");
    println!("  /stages                   - List all available stages");
    println!("  /stage_files <stage>      - List files within a stage");
    println!("  /parse <@stage/file>      - Parse a staged document via AI_PARSE_DOCUMENT");
    println!("  /documents                - List parsed Cortex documents");
    println!("  /download_document <name> - Download a Cortex document as base64");
    println!("  /use_database <db_name>   - Switch to a different database");
    println!("  /use_schema <schema_name> - Switch to a different schema");
    println!("  /use_warehouse <wh_name>  - Switch to a different warehouse");
    println!("  /current                  - Show current database/schema");
    println!("  /text2sql <prompt>        - Generate SQL from natural language");
    println!("  /context                  - Show database context with table counts");
    println!("  /stats                    - Show conversation statistics");
    println!("  /jobs [--clear]           - List document processing jobs");
    println!("  /cancel <job_id>          - Cancel a document processing job");
    println!("  /timeout [seconds]        - Get or set operation timeout");
    println!("  /tools                    - List available AI tools");
    println!("  /exit                     - Exit chat");
    println!();
}

async fn current_database_name(engine: &SnowflakeEngine) -> String {
    engine
        .get_database()
        .await
        .unwrap_or_else(|_| "UNKNOWN".to_string())
}

async fn current_schema_name(engine: &SnowflakeEngine) -> String {
    engine
        .get_schema()
        .await
        .unwrap_or_else(|_| "UNKNOWN".to_string())
}

async fn current_user_name(engine: &SnowflakeEngine) -> String {
    engine
        .get_user()
        .await
        .unwrap_or_else(|_| "UNKNOWN".to_string())
}

async fn handle_common_slash_commands(input: &str, engine: &mut SnowflakeEngine) -> bool {
    if input == "/current" {
        let user = current_user_name(&*engine).await;
        let database = current_database_name(&*engine).await;
        let schema = current_schema_name(&*engine).await;
        println!("\n📍 Current Context:");
        println!("  User: {}", user);
        println!("  Database: {}", database);
        println!("  Schema: {}", schema);
        println!();
        return true;
    }

    if input == "/databases" {
        let current_db = current_database_name(&*engine).await;
        match engine.list_databases().await {
            Ok(databases) => {
                println!("\n📚 Accessible Databases:");
                for db in databases {
                    let marker = if db == current_db { " (current)" } else { "" };
                    println!("  - {}{}", db, marker);
                }
                println!();
            }
            Err(e) => println!("✗ Failed to list databases: {}\n", e),
        }
        return true;
    }

    if input.starts_with("/schemas") {
        let target_db = if input.starts_with("/schemas ") {
            Some(input.strip_prefix("/schemas ").unwrap().trim())
        } else {
            None
        };

        let fallback_db = current_database_name(&*engine).await;
        let current_schema = current_schema_name(&*engine).await;

        match engine.list_schemas(target_db).await {
            Ok(schemas) => {
                let db_name = target_db.unwrap_or(&fallback_db);
                println!("\n📂 Schemas in {}:", db_name);
                for s in schemas {
                    let marker = if s == current_schema && target_db.is_none() {
                        " (current)"
                    } else {
                        ""
                    };
                    println!("  - {}{}", s, marker);
                }
                println!();
            }
            Err(e) => println!("✗ Failed to list schemas: {}\n", e),
        }
        return true;
    }

    if input == "/warehouses" {
        match engine.list_warehouses().await {
            Ok(warehouses) => {
                println!("\n🏢 Accessible Warehouses:");
                for wh in warehouses {
                    println!("  - {}", wh);
                }
                println!();
            }
            Err(e) => println!("✗ Failed to list warehouses: {}\n", e),
        }
        return true;
    }

    if input == "/tables" {
        let current_db = current_database_name(&*engine).await;
        let current_schema = current_schema_name(&*engine).await;
        match engine.get_table_list().await {
            Ok(tables) => {
                println!("\n📋 Tables in {}.{}:", current_db, current_schema);
                for table in tables {
                    println!("  - {}", table);
                }
                println!();
            }
            Err(e) => println!("✗ Failed to list tables: {}\n", e),
        }
        return true;
    }

    if input == "/all_tables" {
        println!("\n🔍 Fetching all accessible tables (this may take a moment)...\n");
        match engine.list_all_tables().await {
            Ok(tables) => {
                let mut current_db = String::new();
                let mut current_schema = String::new();
                for (db, schema, table) in tables {
                    if db != current_db {
                        current_db = db.clone();
                        current_schema.clear();
                        println!("\n{}:", db);
                    }
                    if schema != current_schema {
                        current_schema = schema.clone();
                        println!("  {}:", schema);
                    }
                    println!("    - {}", table);
                }
                println!();
            }
            Err(e) => println!("✗ Failed to list all tables: {}\n", e),
        }
        return true;
    }

    if input.starts_with("/describe ") {
        let table_name = input.strip_prefix("/describe ").unwrap().trim();
        if !table_name.is_empty() {
            let current_db = current_database_name(&*engine).await;
            let current_schema = current_schema_name(&*engine).await;
            match engine.get_table_info(table_name).await {
                Ok(columns) => {
                    println!("Table: {}.{}.{}", current_db, current_schema, table_name);
                    println!("{:<30} {:<20} {:<10}", "Column", "Type", "Nullable");
                    println!("{:-<62}", "");
                    let table_info: serde_json::Value = match serde_json::from_str(&columns) {
                        Ok(info) => info,
                        Err(e) => {
                            println!("Failed to parse table info JSON: {}", e);
                            return true; // Indicate that the command was handled, but with an error
                        }
                    };

                    let Some(cols_to_print) = table_info["columns"].as_array() else {
                        println!("Failed to get columns array from table info");
                        return true; // Indicate that the command was handled, but with an error
                    };

                    println!("Table: {}.{}.{}", current_db, current_schema, table_name);
                    println!("{:<30} {:<20} {:<10}", "Column", "Type", "Nullable");
                    println!("{:-<62}", "");
                    for col in cols_to_print {
                        let column_name = col["COLUMN_NAME"].as_str().unwrap_or("N/A");
                        let data_type = col["DATA_TYPE"].as_str().unwrap_or("N/A");
                        let is_nullable = col["IS_NULLABLE"].as_str().unwrap_or("N/A");
                        println!("{:<30} {:<20} {:<10}", column_name, data_type, is_nullable);
                    }
                }
                Err(e) => println!("Failed to describe table: {}", e),
            }
        } else {
            println!("✗ Usage: /describe <table_name>\n");
        }
        return true;
    }

    if input.starts_with("/complete ") {
        let partial_query = input.strip_prefix("/complete ").unwrap().trim();
        if !partial_query.is_empty() {
            println!("🤖 Completing query (without full context for speed)...");
            match engine.get_query_suggestions(partial_query).await {
                Ok(suggestions) => {
                    if suggestions.is_empty() {
                        println!("✗ No suggestions returned.\n");
                    } else {
                        println!("\n🤖 Suggested Query:\n```sql");
                        for suggestion in suggestions {
                            println!("{}", suggestion);
                        }
                        println!("```");
                    }
                }
                Err(e) => println!("✗ Failed to get suggestions: {}\n", e),
            }
        } else {
            println!("✗ Usage: /complete <partial_query>\n");
        }
        return true;
    }

    if input.starts_with("/suggest_query ") {
        let natural_language_query = input.strip_prefix("/suggest_query ").unwrap().trim();
        if !natural_language_query.is_empty() {
            match engine.get_query_suggestions(natural_language_query).await {
                Ok(suggestions) => {
                    if suggestions.is_empty() {
                        println!("✗ No suggestions found.\n");
                    } else {
                        println!("\n🤖 Suggested Query:\n```sql");
                        for suggestion in suggestions {
                            println!("{}", suggestion);
                        }
                        println!("```");
                    }
                }
                Err(e) => println!("✗ Failed to get suggestions: {}\n", e),
            }
        } else {
            println!("✗ Usage: /suggest_query <natural_language_query>\n");
        }
        return true;
    }

    if input.starts_with("/use_database ") {
        let db_name = input.strip_prefix("/use_database ").unwrap().trim();
        if !db_name.is_empty() {
            match engine.use_database(db_name).await {
                Ok(_) => println!("✓ Switched to database: {}\n", db_name),
                Err(e) => println!("✗ Failed to switch database: {}\n", e),
            }
        }
        return true;
    }

    if input.starts_with("/use_warehouse ") {
        let wh_name = input.strip_prefix("/use_warehouse ").unwrap().trim();
        if !wh_name.is_empty() {
            match engine.use_warehouse(wh_name).await {
                Ok(_) => println!("✓ Switched to warehouse: {}\n", wh_name),
                Err(e) => println!("✗ Failed to switch warehouse: {}\n", e),
            }
        }
        return true;
    }

    if input.starts_with("/use_schema ") {
        let schema_name = input.strip_prefix("/use_schema ").unwrap().trim();
        if !schema_name.is_empty() {
            match engine.use_schema(schema_name).await {
                Ok(_) => println!("✓ Switched to schema: {}\n", schema_name),
                Err(e) => println!("✗ Failed to switch schema: {}\n", e),
            }
        }
        return true;
    }

    false
}

#[derive(Debug, Clone)]
struct ConversationSummary {
    id: String,
    label: Option<String>,
    message_count: usize,
    updated_at: Option<String>,
}

#[derive(Debug, Clone)]
struct ConversationConfig {
    label: Option<String>,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
}

#[derive(Debug, Default, Clone, Copy)]
struct ConversationOverrides {
    label: bool,
    model: bool,
    temperature: bool,
    max_tokens: bool,
}

impl ConversationConfig {
    fn to_metadata(&self) -> Value {
        json!({
            "label": self.label,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        })
    }

    fn absorb_metadata(&mut self, metadata: &Value, overrides: &ConversationOverrides) {
        if !overrides.label {
            if let Some(label) = metadata.get("label").and_then(|v| v.as_str()) {
                self.label = Some(label.to_string());
            }
        }

        if !overrides.model {
            if let Some(model) = metadata.get("model").and_then(|v| v.as_str()) {
                self.model = model.to_string();
            }
        }

        if !overrides.temperature {
            if let Some(temp) = metadata.get("temperature").and_then(|v| v.as_f64()) {
                self.temperature = Some(temp as f32);
            }
        }

        if !overrides.max_tokens {
            if let Some(tokens) = metadata.get("max_tokens").and_then(|v| v.as_i64()) {
                self.max_tokens = Some(tokens as i32);
            }
        }
    }
}

const MODEL_PREFERENCE_KEY: &str = "model_preference";
const TEMPERATURE_PREFERENCE_KEY: &str = "temperature_preference";
const MAX_TOKENS_PREFERENCE_KEY: &str = "max_tokens_preference";
const LABEL_PREFERENCE_KEY: &str = "conversation_label";
const DATABASE_PREFERENCE_KEY: &str = "database_preference";
const SCHEMA_PREFERENCE_KEY: &str = "schema_preference";

fn normalize_context_string(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn to_context_payload(value: &str) -> String {
    if serde_json::from_str::<Value>(value).is_ok() {
        value.replace("$$", "$ $")
    } else {
        json!({ "value": value }).to_string().replace("$$", "$ $")
    }
}

async fn fetch_context_map(
    engine: &SnowflakeEngine,
    conversation_id: &str,
) -> Result<HashMap<String, String>, String> {
    let user_name = engine.get_user().await?;
    engine.setup_user_resources(&user_name).await?;
    let schema = engine.user_schema_name();
    let escaped_id = sanitize_sql_string(conversation_id);
    let query = format!(
        "SELECT OBJECT_KEY, COALESCE(OBJECT_VALUE:value::STRING, TO_JSON(OBJECT_VALUE)::STRING) AS VALUE \
         FROM OPENBB_AGENTS.{schema}.AGENTS_CONTEXT_OBJECTS \
         WHERE CONVERSATION_ID = '{escaped_id}'"
    );
    let rows = engine.execute_statement(&query).await?;
    let mut map = HashMap::new();
    if let Some(array) = rows.as_array() {
        for row in array {
            if let Some(key) = row
                .get("OBJECT_KEY")
                .or_else(|| row.get("object_key"))
                .and_then(|v| v.as_str())
            {
                if let Some(value) = row
                    .get("VALUE")
                    .or_else(|| row.get("value"))
                    .and_then(|v| v.as_str())
                {
                    if let Some(normalized) = normalize_context_string(value) {
                        map.insert(key.to_string(), normalized);
                    }
                }
            }
        }
    }
    Ok(map)
}

/// Context preferences to restore when loading a conversation
struct RestoredContext {
    database: Option<String>,
    schema: Option<String>,
}

async fn apply_context_preferences(
    engine: &SnowflakeEngine,
    conversation_id: &str,
    config: &mut ConversationConfig,
    overrides: &ConversationOverrides,
) -> Result<RestoredContext, String> {
    let context_map = fetch_context_map(engine, conversation_id).await?;

    if !overrides.model {
        if let Some(model) = context_map.get(MODEL_PREFERENCE_KEY) {
            if !model.trim().is_empty() {
                config.model = model.to_string();
            }
        }
    }

    if !overrides.temperature {
        if let Some(temp_str) = context_map.get(TEMPERATURE_PREFERENCE_KEY) {
            if let Ok(value) = temp_str.parse::<f32>() {
                config.temperature = Some(value);
            }
        }
    }

    if !overrides.max_tokens {
        if let Some(tokens_str) = context_map.get(MAX_TOKENS_PREFERENCE_KEY) {
            if let Ok(value) = tokens_str.parse::<i32>() {
                config.max_tokens = Some(value);
            }
        }
    }

    if !overrides.label {
        if let Some(label) = context_map.get(LABEL_PREFERENCE_KEY) {
            if !label.trim().is_empty() {
                config.label = Some(label.to_string());
            }
        }
    }

    // Extract database/schema preferences to be applied by caller
    let database = context_map
        .get(DATABASE_PREFERENCE_KEY)
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string());
    let schema = context_map
        .get(SCHEMA_PREFERENCE_KEY)
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.to_string());

    Ok(RestoredContext { database, schema })
}

async fn set_context_value(
    engine: &SnowflakeEngine,
    conversation_id: &str,
    key: &str,
    value: &str,
) -> Result<(), String> {
    let user_name = engine.get_user().await?;
    engine.setup_user_resources(&user_name).await?;
    let schema = engine.user_schema_name();
    let object_id = format!("{}_{}", conversation_id, key);
    let escaped_object_id = sanitize_sql_string(&object_id);
    let escaped_conv = sanitize_sql_string(conversation_id);
    let escaped_key = sanitize_sql_string(key);
    let payload = to_context_payload(value);
    let query = format!(
        "MERGE INTO OPENBB_AGENTS.{schema}.AGENTS_CONTEXT_OBJECTS AS target \
         USING (SELECT 1) AS source \
         ON target.OBJECT_ID = '{escaped_object_id}' \
         WHEN MATCHED THEN UPDATE SET OBJECT_VALUE = PARSE_JSON($${payload}$$) \
         WHEN NOT MATCHED THEN INSERT (OBJECT_ID, CONVERSATION_ID, OBJECT_TYPE, OBJECT_KEY, OBJECT_VALUE) \
         VALUES ('{escaped_object_id}', '{escaped_conv}', 'context', '{escaped_key}', PARSE_JSON($${payload}$$))"
    );
    engine.execute_statement(&query).await.map(|_| ())
}

async fn delete_context_value(
    engine: &SnowflakeEngine,
    conversation_id: &str,
    key: &str,
) -> Result<(), String> {
    let user_name = engine.get_user().await?;
    engine.setup_user_resources(&user_name).await?;
    let schema = engine.user_schema_name();
    let object_id = format!("{}_{}", conversation_id, key);
    let escaped_object_id = sanitize_sql_string(&object_id);
    let query = format!(
        "DELETE FROM OPENBB_AGENTS.{schema}.AGENTS_CONTEXT_OBJECTS WHERE OBJECT_ID = '{escaped_object_id}'"
    );
    engine.execute_statement(&query).await.map(|_| ())
}

async fn persist_context_preferences(
    engine: &SnowflakeEngine,
    conversation_id: &str,
    config: &ConversationConfig,
) -> Result<(), String> {
    set_context_value(engine, conversation_id, MODEL_PREFERENCE_KEY, &config.model).await?;

    if let Some(temp) = config.temperature {
        set_context_value(
            engine,
            conversation_id,
            TEMPERATURE_PREFERENCE_KEY,
            &temp.to_string(),
        )
        .await?;
    } else {
        let _ = delete_context_value(engine, conversation_id, TEMPERATURE_PREFERENCE_KEY).await;
    }

    if let Some(tokens) = config.max_tokens {
        set_context_value(
            engine,
            conversation_id,
            MAX_TOKENS_PREFERENCE_KEY,
            &tokens.to_string(),
        )
        .await?;
    } else {
        let _ = delete_context_value(engine, conversation_id, MAX_TOKENS_PREFERENCE_KEY).await;
    }

    match config
        .label
        .as_ref()
        .map(|label| label.trim())
        .filter(|s| !s.is_empty())
    {
        Some(label) => {
            set_context_value(engine, conversation_id, LABEL_PREFERENCE_KEY, label).await?;
        }
        None => {
            let _ = delete_context_value(engine, conversation_id, LABEL_PREFERENCE_KEY).await;
        }
    }

    Ok(())
}

/// Save the current database/schema context to the conversation
async fn save_database_context(
    engine: &SnowflakeEngine,
    conversation_id: &str,
) -> Result<(), String> {
    let database = engine.get_database().await.unwrap_or_default();
    let schema = engine.get_schema().await.unwrap_or_default();

    if !database.is_empty() {
        set_context_value(engine, conversation_id, DATABASE_PREFERENCE_KEY, &database).await?;
    }
    if !schema.is_empty() {
        set_context_value(engine, conversation_id, SCHEMA_PREFERENCE_KEY, &schema).await?;
    }

    Ok(())
}

fn sanitize_sql_string(input: &str) -> String {
    input.replace('\'', "''")
}

async fn print_available_models(engine: &SnowflakeEngine) {
    println!("\n📊 Available Models:");
    match engine.get_available_models().await {
        Ok(models) => {
            for (name, desc) in models {
                println!("- {:<25} {}", name, desc);
            }
            println!("  (Use /model <name> to switch)\n");
        }
        Err(err) => {
            println!("✗ Failed to fetch available models: {}\n", err);
        }
    }
}

type StdinLines = tokio::io::Lines<tokio::io::BufReader<tokio::io::Stdin>>;

fn expand_path(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return Path::new(&home).join(stripped);
        }
    }
    if path == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(path)
}

fn get_value_case_insensitive(row: &Value, key: &str) -> Option<String> {
    if let Value::Object(map) = row {
        if let Some(val) = map.get(key) {
            return val.as_str().map(|s| s.to_string());
        }
        let upper = key.to_uppercase();
        if let Some(val) = map.get(&upper) {
            return val.as_str().map(|s| s.to_string());
        }
        let lower = key.to_lowercase();
        if let Some(val) = map.get(&lower) {
            return val.as_str().map(|s| s.to_string());
        }
    }
    None
}

fn extract_last_sql_snippet(text: &str) -> Option<String> {
    let marker = "```sql";
    let start = text.rfind(marker)?;
    let after_marker = &text[start + marker.len()..];
    let end_offset = after_marker.find("```")?;
    let snippet = after_marker[..end_offset].trim();
    if snippet.is_empty() {
        None
    } else {
        Some(snippet.to_string())
    }
}

async fn prompt_for_sql_input(lines: &mut StdinLines) -> Option<String> {
    println!("Enter SQL (terminate with ';' and an empty line):");
    let mut buffer = String::new();
    loop {
        print!("sql> ");
        let _ = std::io::stdout().flush();
        match lines.next_line().await {
            Ok(Some(line)) => {
                let trimmed = line.trim_end().to_string();
                if trimmed.is_empty() && buffer.trim_end().ends_with(';') {
                    break;
                }
                buffer.push_str(&trimmed);
                buffer.push('\n');
            }
            Ok(None) => break,
            Err(_) => return None,
        }
    }
    let cleaned = buffer.trim().trim_end_matches(';').trim().to_string();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// Prompt for SQL input using rustyline editor
fn prompt_for_sql_input_rl<H: Helper>(
    rl: &mut Editor<H, rustyline::history::DefaultHistory>,
) -> Option<String> {
    println!("Enter SQL (terminate with ';' and an empty line):");
    let mut buffer = String::new();
    loop {
        match rl.readline("sql> ") {
            Ok(line) => {
                let trimmed = line.trim_end().to_string();
                if trimmed.is_empty() && buffer.trim_end().ends_with(';') {
                    break;
                }
                buffer.push_str(&trimmed);
                buffer.push('\n');
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C - Cancelled");
                return None;
            }
            Err(ReadlineError::Eof) => break,
            Err(_) => return None,
        }
    }
    let cleaned = buffer.trim().trim_end_matches(';').trim().to_string();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

async fn handle_document_commands(
    input: &str,
    engine: &mut SnowflakeEngine,
    job_manager: &mut JobManager,
    emitter: &EventEmitter,
) -> bool {
    if input == "/stages" {
        match engine.execute_statement("SHOW STAGES").await {
            Ok(rows) => {
                let mut stages = BTreeSet::new();
                if let Some(array) = rows.as_array() {
                    let current_db = current_database_name(engine).await;
                    let current_schema = current_schema_name(engine).await;
                    for row in array {
                        let name = get_value_case_insensitive(row, "NAME");
                        let db =
                            get_value_case_insensitive(row, "DATABASE_NAME").unwrap_or_default();
                        let schema =
                            get_value_case_insensitive(row, "SCHEMA_NAME").unwrap_or_default();
                        if let Some(stage_name) = name {
                            if db.is_empty() || schema.is_empty() {
                                stages.insert(stage_name);
                            } else if db == current_db && schema == current_schema {
                                stages.insert(stage_name);
                            } else {
                                stages.insert(format!("{}.{}.{}", db, schema, stage_name));
                            }
                        }
                    }
                }

                if let Ok(user) = engine.get_user().await {
                    let sanitized: String = user
                        .chars()
                        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                        .collect();
                    let user_schema = format!("USER_{}", sanitized).to_uppercase();
                    if !stages.iter().any(|s| s.contains(&user_schema)) {
                        let query = format!("SHOW STAGES IN SCHEMA OPENBB_AGENTS.{}", user_schema);
                        if let Ok(user_rows) = engine.execute_statement(&query).await {
                            if let Some(array) = user_rows.as_array() {
                                for row in array {
                                    if let Some(name) = get_value_case_insensitive(row, "NAME") {
                                        stages.insert(format!(
                                            "OPENBB_AGENTS.{}.{}",
                                            user_schema, name
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }

                if stages.is_empty() {
                    println!("No stages found.\n");
                } else {
                    println!("\n📁 Available Stages:");
                    for stage in stages {
                        println!("- {}", stage);
                    }
                    println!();
                }
            }
            Err(err) => println!("✗ Failed to list stages: {}\n", err),
        }
        return true;
    }

    if let Some(stage_name) = input.strip_prefix("/stage_files ") {
        let stage_name = stage_name.trim();
        if stage_name.is_empty() {
            println!("✗ Usage: /stage_files <stage_name>\n");
            return true;
        }

        let list_target = if stage_name.starts_with('@') {
            format!("LIST {}", stage_name)
        } else if stage_name.contains('.') {
            format!("LIST @{}", stage_name)
        } else {
            format!("LIST @{}", stage_name)
        };

        match engine.execute_statement(&list_target).await {
            Ok(rows) => {
                if let Some(array) = rows.as_array() {
                    if array.is_empty() {
                        println!("No files found in stage {}.\n", stage_name);
                    } else {
                        println!("\n📄 Files in {}:", stage_name);
                        for row in array {
                            if let Some(name) = get_value_case_insensitive(row, "NAME") {
                                let filename = name.split('/').last().unwrap_or(&name).to_string();
                                println!("- {}", filename);
                            }
                        }
                        println!();
                    }
                }
            }
            Err(err) => println!("✗ Failed to list files: {}\n", err),
        }
        return true;
    }

    if let Some(path_str) = input.strip_prefix("/upload ") {
        let trimmed = path_str.trim();
        if trimmed.is_empty() {
            println!("✗ Usage: /upload <file_path>\n");
            return true;
        }
        let path = expand_path(trimmed);
        if !path.exists() || !path.is_file() {
            println!("✗ File not found: {}\n", path.display());
            return true;
        }
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("upload.bin")
            .to_string();

        // Create job for tracking
        let (job_id, cancel_token) = job_manager.create_job(&file_name);

        // Ensure jobs table exists
        if let Err(e) = jobs::ensure_jobs_table(engine).await {
            println!("✗ Failed to initialize jobs table: {}\n", e);
            return true;
        }

        println!("📤 Starting upload job {} for {}", &job_id[..8], file_name);
        emitter.job_status(
            &job_id,
            &file_name,
            "pending",
            Some("Reading file..."),
            None,
        );

        // Read file
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(err) => {
                println!("✗ Failed to read file: {}\n", err);
                job_manager.remove_job(&job_id);
                return true;
            }
        };

        // Check for cancellation
        if cancel_token.is_cancelled() {
            println!("⊘ Upload cancelled.\n");
            job_manager.remove_job(&job_id);
            return true;
        }

        emitter.job_status(
            &job_id,
            &file_name,
            "processing",
            Some("Uploading to stage..."),
            None,
        );

        // Upload to stage
        match engine.upload_bytes_to_stage(&bytes, &file_name, None).await {
            Ok(stage_path) => {
                // Insert job record in database
                if let Err(e) = jobs::insert_job(engine, &job_id, &file_name, &stage_path).await {
                    println!("⚠ Upload succeeded but failed to record job: {}", e);
                }

                // Update job status to complete
                let _ = jobs::update_job_status(
                    engine,
                    &job_id,
                    jobs::JobStatus::Complete,
                    Some("Upload complete"),
                    None,
                )
                .await;

                println!("✓ Uploaded {} to {}", file_name, stage_path);
                println!("  Job ID: {}", job_id);
                println!("  Use /parse {} to process the document.\n", stage_path);

                emitter.job_status(
                    &job_id,
                    &file_name,
                    "complete",
                    Some("Upload complete"),
                    None,
                );
            }
            Err(err) => {
                // Record failure
                let _ = jobs::insert_job(engine, &job_id, &file_name, "").await;
                let _ = jobs::update_job_status(
                    engine,
                    &job_id,
                    jobs::JobStatus::Failed,
                    None,
                    Some(&err),
                )
                .await;

                println!("✗ Upload failed: {}\n", err);
                emitter.job_status(&job_id, &file_name, "failed", None, Some(&err));
            }
        }

        job_manager.remove_job(&job_id);
        return true;
    }

    if let Some(stage_path) = input.strip_prefix("/parse ") {
        let stage_path = stage_path.trim();
        if stage_path.is_empty() {
            println!("✗ Usage: /parse <stage_path>\n");
            return true;
        }

        // Extract filename from stage path for display
        let file_name = stage_path
            .split('/')
            .last()
            .unwrap_or(stage_path)
            .to_string();

        // Create job for tracking
        let (job_id, cancel_token) = job_manager.create_job(&file_name);

        // Ensure jobs table exists
        if let Err(e) = jobs::ensure_jobs_table(engine).await {
            println!("✗ Failed to initialize jobs table: {}\n", e);
            return true;
        }

        // Insert job record
        if let Err(e) = jobs::insert_job(engine, &job_id, &file_name, stage_path).await {
            println!("⚠ Failed to record job: {}", e);
        }

        println!("📄 Starting parse job {} for {}", &job_id[..8], file_name);
        emitter.job_status(
            &job_id,
            &file_name,
            "processing",
            Some("Parsing document..."),
            None,
        );

        // Update status to processing
        let _ = jobs::update_job_status(
            engine,
            &job_id,
            jobs::JobStatus::Processing,
            Some("Parsing document..."),
            None,
        )
        .await;

        // Check for cancellation
        if cancel_token.is_cancelled() {
            let _ = jobs::update_job_status(
                engine,
                &job_id,
                jobs::JobStatus::Cancelled,
                None,
                Some("Cancelled by user"),
            )
            .await;
            println!("⊘ Parse cancelled.\n");
            job_manager.remove_job(&job_id);
            return true;
        }

        match engine.ai_parse_document(stage_path, Some("LAYOUT")).await {
            Ok(result) => {
                // Update job status to complete
                let _ = jobs::update_job_status(
                    engine,
                    &job_id,
                    jobs::JobStatus::Complete,
                    Some("Parse complete"),
                    None,
                )
                .await;

                println!(
                    "✓ Parsed document. Job ID: {}\n  Result preview (first 500 chars):\n{}\n",
                    job_id,
                    &result.chars().take(500).collect::<String>()
                );

                emitter.job_status(
                    &job_id,
                    &file_name,
                    "complete",
                    Some("Parse complete"),
                    None,
                );
            }
            Err(err) => {
                // Record failure
                let _ = jobs::update_job_status(
                    engine,
                    &job_id,
                    jobs::JobStatus::Failed,
                    None,
                    Some(&err),
                )
                .await;

                println!("✗ Document parsing failed: {}\n", err);
                emitter.job_status(&job_id, &file_name, "failed", None, Some(&err));
            }
        }

        job_manager.remove_job(&job_id);
        return true;
    }

    if input == "/documents" {
        match engine.list_cortex_documents().await {
            Ok(documents) => {
                if documents.is_empty() {
                    println!("No Cortex documents found.\n");
                } else {
                    println!("\n📚 Cortex Documents:");
                    for (file_name, stage_path, parsed, page_count) in documents {
                        println!(
                            "- {} (stage: {}, parsed: {}, pages: {})",
                            file_name, stage_path, parsed, page_count
                        );
                    }
                    println!();
                }
            }
            Err(err) => println!("✗ Failed to list documents: {}\n", err),
        }
        return true;
    }

    if let Some(doc_name) = input.strip_prefix("/download_document ") {
        let doc_name = doc_name.trim();
        if doc_name.is_empty() {
            println!("✗ Usage: /download_document <file_name>\n");
            return true;
        }
        match engine.download_cortex_document(doc_name).await {
            Ok(content) => {
                println!(
                    "✓ Downloaded document {} ({} bytes base64).\n",
                    doc_name,
                    content.len()
                );
            }
            Err(err) => println!("✗ Failed to download document: {}\n", err),
        }
        return true;
    }

    false
}

async fn fetch_conversation_summaries(
    engine: &SnowflakeEngine,
) -> Result<Vec<ConversationSummary>, String> {
    let user_name = engine.get_user().await?;
    engine.setup_user_resources(&user_name).await?;
    let schema = engine.user_schema_name();
    let query = format!(
        "SELECT CONVERSATION_ID, \
                COALESCE(TO_CHAR(LAST_UPDATED_AT, 'YYYY-MM-DD HH24:MI:SS'), \
                         TO_CHAR(CREATED_AT, 'YYYY-MM-DD HH24:MI:SS')) AS UPDATED_AT, \
                METADATA:label::STRING AS LABEL, \
                COALESCE((SELECT COUNT(*) FROM OPENBB_AGENTS.{schema}.AGENTS_MESSAGES m \
                          WHERE m.CONVERSATION_ID = c.CONVERSATION_ID), 0) AS MESSAGE_COUNT \
         FROM OPENBB_AGENTS.{schema}.AGENTS_CONVERSATIONS c \
         ORDER BY COALESCE(LAST_UPDATED_AT, CREATED_AT) DESC"
    );

    let rows = engine.execute_statement(&query).await?;
    let mut summaries = Vec::new();
    let mut seen_ids = HashSet::new();
    if let Some(array) = rows.as_array() {
        for row in array {
            let conversation_id = row
                .get("CONVERSATION_ID")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            if conversation_id.is_empty() {
                continue;
            }

            if !seen_ids.insert(conversation_id.clone()) {
                continue;
            }

            let label = row
                .get("LABEL")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let updated_at = row
                .get("UPDATED_AT")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let message_count = row
                .get("MESSAGE_COUNT")
                .map(|value| match value {
                    Value::Number(num) => num.as_u64().unwrap_or(0) as usize,
                    Value::String(text) => text.parse::<usize>().unwrap_or(0),
                    _ => 0,
                })
                .unwrap_or(0);

            summaries.push(ConversationSummary {
                id: conversation_id,
                label,
                updated_at,
                message_count,
            });
        }
    }

    Ok(summaries)
}

fn print_conversation_summaries(summaries: &[ConversationSummary]) {
    if summaries.is_empty() {
        println!("No stored conversations found.\n");
        return;
    }

    println!("\n📂 Conversations:");
    println!(
        "{:<40} {:<20} {:<10} {}",
        "Conversation ID", "Last Updated", "Messages", "Label"
    );

    for summary in summaries {
        println!(
            "{:<40} {:<20} {:<10} {}",
            summary.id,
            summary.updated_at.as_deref().unwrap_or("-"),
            summary.message_count,
            summary.label.as_deref().unwrap_or("(unlabeled)")
        );
    }
    println!();
}

async fn refresh_conversation_history(
    engine: &SnowflakeEngine,
    conversation_id: &str,
) -> Result<Vec<agents::Message>, String> {
    let user_name = engine.get_user().await?;
    engine.setup_user_resources(&user_name).await?;
    let schema = engine.user_schema_name();
    let escaped_id = sanitize_sql_string(conversation_id);
    let query = format!(
        "SELECT ROLE, CONTENT FROM OPENBB_AGENTS.{schema}.AGENTS_MESSAGES \
         WHERE CONVERSATION_ID = '{escaped_id}' \
         ORDER BY TIMESTAMP ASC"
    );
    let rows = engine.execute_statement(&query).await?;
    let mut history = Vec::new();
    if let Some(array) = rows.as_array() {
        for row in array {
            let role = row
                .get("ROLE")
                .and_then(|v| v.as_str())
                .unwrap_or("assistant")
                .to_string();
            let content = row
                .get("CONTENT")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let message = match role.as_str() {
                "system" => agents::Message::new_system(content),
                "assistant" => agents::Message::new_assistant(content),
                _ => agents::Message::new_user(content),
            };
            history.push(message);
        }
    }
    Ok(history)
}

async fn persist_conversation_message(
    engine: &SnowflakeEngine,
    conversation_id: &str,
    role: &str,
    content: &str,
) -> Result<(), String> {
    let user_name = engine.get_user().await?;
    engine.setup_user_resources(&user_name).await?;
    let schema = engine.user_schema_name();
    let escaped_id = sanitize_sql_string(conversation_id);
    let escaped_content = sanitize_sql_string(content);
    let escaped_role = sanitize_sql_string(role);
    let message_id = Uuid::new_v4().to_string();
    let query = format!(
        "INSERT INTO OPENBB_AGENTS.{schema}.AGENTS_MESSAGES \
         (MESSAGE_ID, CONVERSATION_ID, ROLE, CONTENT) \
         VALUES ('{message_id}', '{escaped_id}', '{escaped_role}', '{escaped_content}')"
    );
    engine.execute_statement(&query).await?;

    let touch_query = format!(
        "UPDATE OPENBB_AGENTS.{schema}.AGENTS_CONVERSATIONS \
         SET LAST_UPDATED_AT = CURRENT_TIMESTAMP() \
         WHERE CONVERSATION_ID = '{escaped_id}'"
    );
    engine.execute_statement(&touch_query).await?;
    Ok(())
}

async fn clear_conversation_messages(
    engine: &SnowflakeEngine,
    conversation_id: &str,
) -> Result<(), String> {
    let user_name = engine.get_user().await?;
    engine.setup_user_resources(&user_name).await?;
    let schema = engine.user_schema_name();
    let escaped_id = sanitize_sql_string(conversation_id);
    let delete_messages = format!(
        "DELETE FROM OPENBB_AGENTS.{schema}.AGENTS_MESSAGES WHERE CONVERSATION_ID = '{escaped_id}'"
    );
    engine.execute_statement(&delete_messages).await?;

    let delete_context = format!(
        "DELETE FROM OPENBB_AGENTS.{schema}.AGENTS_CONTEXT_OBJECTS WHERE CONVERSATION_ID = '{escaped_id}'"
    );
    engine.execute_statement(&delete_context).await?;

    let refresh_convo = format!(
        "UPDATE OPENBB_AGENTS.{schema}.AGENTS_CONVERSATIONS \
         SET LAST_UPDATED_AT = CURRENT_TIMESTAMP(), SUMMARY = NULL \
         WHERE CONVERSATION_ID = '{escaped_id}'"
    );
    engine.execute_statement(&refresh_convo).await?;
    Ok(())
}

async fn sync_conversation_metadata(
    engine: &SnowflakeEngine,
    conversation_id: &str,
    config: &ConversationConfig,
) -> Result<(), String> {
    engine
        .update_conversation_settings(conversation_id, config.to_metadata())
        .await?;
    persist_context_preferences(engine, conversation_id, config).await
}

async fn build_system_prompt(engine: &SnowflakeEngine, tool_executor: &ToolExecutor) -> String {
    let database = engine
        .get_database()
        .await
        .unwrap_or_else(|_| "UNKNOWN".to_string());
    let schema = engine
        .get_schema()
        .await
        .unwrap_or_else(|_| "UNKNOWN".to_string());
    let user_schema = engine.user_schema_name();

    // Get tool JSON definitions for the LLM
    let tools_json = tool_executor.registry().get_tools_json();
    let tools_json_str = serde_json::to_string_pretty(&tools_json).unwrap_or_default();

    format!(
        r#"You are an AI assistant with a specific focus on the Snowflake dialect of SQL. Your goal is to help the user by directly answering their questions using the available tools.

🚨 CRITICAL: NEVER MAKE UP DATA OR EMBELLISH TOOL RESULTS
- Tool results contain REAL data from Snowflake - display it EXACTLY as returned
- DO NOT add fake descriptions, made-up explanations, or embellished details
- Your job is to PRESENT data accurately, not to CREATE fictional content

DATABASE CONTEXT:
- Current database: {database}
- Current schema: {schema}
- User schema: OPENBB_AGENTS.{user_schema}

🚨 TOOL SELECTION:

FOR GETTING DATA / ANSWERING QUESTIONS:
- Use execute_query - write SQL yourself and execute it to get results
- Use list_tables_in - to see what tables exist in a schema  
- Use list_semantic_views - to see what semantic views exist
- Use get_table_schema - to see column definitions
- Use get_table_sample_data - to see sample rows

FOR GENERATING SQL CODE AS OUTPUT (user wants the SQL, not results):
- Use text2sql ONLY when user says "write me a query", "generate SQL", "create a query for me"
- text2sql returns SQL code for the user to review - it does NOT execute anything

FOR RENDERING CHARTS:
- Use render_chart AFTER execute_query returns data
- You MUST pass the actual data array from the query result, NOT just column names

SNOWFLAKE SQL SYNTAX:
- ALWAYS use double quotes ("") for column names to preserve exact case
- ALWAYS use fully qualified table names: DATABASE.SCHEMA.TABLE
- Snowflake converts unquoted identifiers to UPPERCASE

AVAILABLE TOOLS (JSON SCHEMA):
{tools_json_str}

TO CALL A TOOL, output ONLY this exact JSON format (no other text before or after):
{{"tool": "<tool_name>", "arguments": {{<args>}}}}

Example: {{"tool": "list_schemas", "arguments": {{"database": "MY_DB"}}}}
Example: {{"tool": "list_databases", "arguments": {{}}}}
Example: {{"tool": "execute_query", "arguments": {{"sql": "SELECT * FROM MY_TABLE LIMIT 10"}}}}

CRITICAL: When the user asks for data, OUTPUT THE TOOL CALL JSON IMMEDIATELY. Do not ask questions or explain - just call the tool.
"#
    )
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut env_file_provided = false;

    if let Some(pos) = args
        .iter()
        .position(|arg| arg == "--env-file" || arg == "-e")
    {
        if let Some(path) = args.get(pos + 1) {
            if !path.starts_with('-') {
                dotenvy::from_path(path).expect("Failed to load .env file");
                env_file_provided = true;
            }
        }
    }

    if !env_file_provided {
        if let Ok(home) = std::env::var("HOME") {
            let default_path = Path::new(&home).join(".openbb_platform/.env");
            if default_path.exists() {
                dotenvy::from_path(&default_path).expect("Failed to load default .env file");
            }
        }
    }

    let args = Args::parse();

    match args.command {
        Commands::Lsp {
            user,
            password,
            account,
            role,
            warehouse,
            database,
            schema,
            ..
        } => {
            let engine = SnowflakeEngine::new(
                &user, &password, &account, &role, &warehouse, &database, &schema,
            )
            .await;
            lsp::start_lsp(engine.ok()).await;
        }
        Commands::Execute {
            user,
            password,
            account,
            role,
            warehouse,
            database,
            schema,
            ..
        } => {
            match SnowflakeEngine::new(
                &user, &password, &account, &role, &warehouse, &database, &schema,
            )
            .await
            {
                Ok(engine) => {
                    println!("Connected to Snowflake. Enter queries to execute (end with ';'):");
                    let mut query_buffer = String::new();
                    let mut lines = io::BufReader::new(io::stdin()).lines();

                    while let Ok(Some(line)) = lines.next_line().await {
                        if line.is_empty() && query_buffer.contains(';') {
                            if let Some(pos) = query_buffer.rfind(';') {
                                let query = &query_buffer[..pos];
                                if !query.trim().is_empty() {
                                    match engine.execute_query(query, None, None, None).await {
                                        Ok(results) => println!(
                                            "{}",
                                            serde_json::to_string_pretty(&results).unwrap()
                                        ),
                                        Err(e) => println!("Query failed: {}", e),
                                    }
                                }
                            }
                            query_buffer.clear();
                        } else {
                            query_buffer.push_str(&line);
                            query_buffer.push('\n');
                        }
                    }
                }
                Err(e) => println!("Failed to connect to Snowflake: {}", e),
            }
        }
        Commands::Validate {
            user,
            password,
            account,
            role,
            warehouse,
            database,
            schema,
            ..
        } => {
            match SnowflakeEngine::new(
                &user, &password, &account, &role, &warehouse, &database, &schema,
            )
            .await
            {
                Ok(engine) => {
                    println!("Connected to Snowflake. Enter queries to validate (end with ';'):");
                    let mut query_buffer = String::new();
                    let mut lines = io::BufReader::new(io::stdin()).lines();

                    while let Ok(Some(line)) = lines.next_line().await {
                        if line.is_empty() && query_buffer.contains(';') {
                            if let Some(pos) = query_buffer.rfind(';') {
                                let query = &query_buffer[..pos];
                                if !query.trim().is_empty() {
                                    match engine.validate_query(query).await {
                                        Ok(_) => println!("✓ Query is valid."),
                                        Err(e) => println!("✗ Query is invalid: {}", e),
                                    }
                                }
                            }
                            query_buffer.clear();
                        } else {
                            query_buffer.push_str(&line);
                            query_buffer.push('\n');
                        }
                    }
                }
                Err(e) => println!("Failed to connect to Snowflake: {}", e),
            }
        }
        Commands::Chat {
            user,
            password,
            account,
            role,
            warehouse,
            database,
            schema,
            semantic_model,
            conversation_id,
            conversation_label,
            list_conversations,
            model,
            temperature,
            max_tokens,
            json,
            timeout,
            ..
        } => {
            // Create event emitter based on --json flag
            let emitter = create_emitter(json);

            // Create job manager with configured timeout
            let mut job_manager = JobManager::new();
            job_manager.set_timeout(timeout);

            // Create tool executor
            let tool_executor = ToolExecutor::new();

            match SnowflakeEngine::new(
                &user, &password, &account, &role, &warehouse, &database, &schema,
            )
            .await
            {
                Ok(mut engine) => {
                    if list_conversations {
                        let current_user = match engine.get_user().await {
                            Ok(name) => name,
                            Err(err) => {
                                println!("✗ Failed to resolve user: {}", err);
                                return;
                            }
                        };
                        if let Err(err) = engine.setup_user_resources(&current_user).await {
                            println!("✗ Failed to load conversations: {}", err);
                        } else {
                            match fetch_conversation_summaries(&engine).await {
                                Ok(summaries) => print_conversation_summaries(&summaries),
                                Err(err) => println!("✗ Failed to list conversations: {}", err),
                            }
                        }
                        return;
                    }

                    engine.configure_from_env().await;

                    if let Some(model_path) = semantic_model {
                        match engine
                            .upload_semantic_model(&model_path, "cortex_analyst_semantic_models")
                            .await
                        {
                            Ok(msg) => println!("✓ {}", msg),
                            Err(e) => {
                                println!("✗ Failed to upload semantic model: {}", e);
                                return;
                            }
                        }
                    }

                    let current_user = match engine.get_user().await {
                        Ok(name) => name,
                        Err(err) => {
                            println!("✗ Failed to resolve user: {}", err);
                            return;
                        }
                    };

                    if let Err(err) = engine.setup_user_resources(&current_user).await {
                        println!("✗ Failed to prepare Snowflake resources: {}", err);
                        return;
                    }

                    let mut agent_client = agents::AgentsClient::new(
                        account.clone(),
                        password.clone(),
                        database.clone(),
                        schema.clone(),
                    );

                    let mut active_conversation_id = conversation_id
                        .and_then(|value| {
                            let trimmed = value.trim().to_string();
                            if trimmed.is_empty() {
                                None
                            } else {
                                Some(trimmed)
                            }
                        })
                        .unwrap_or_else(|| Uuid::new_v4().to_string());

                    let trimmed_label = conversation_label.and_then(|value| {
                        let trimmed = value.trim().to_string();
                        if trimmed.is_empty() {
                            None
                        } else {
                            Some(trimmed)
                        }
                    });

                    let model_supplied = model.is_some();
                    let initial_model = model.unwrap_or_else(|| "llama3.1-70b".to_string());
                    let temperature_override = temperature.is_some();
                    let max_tokens_override = max_tokens.is_some();
                    let label_override = trimmed_label.is_some();

                    let mut config = ConversationConfig {
                        label: trimmed_label,
                        model: initial_model,
                        temperature,
                        max_tokens,
                    };

                    let overrides = ConversationOverrides {
                        label: label_override,
                        model: model_supplied,
                        temperature: temperature_override,
                        max_tokens: max_tokens_override,
                    };

                    let metadata_value = config.to_metadata();
                    let stored_metadata = engine
                        .get_or_create_conversation(&active_conversation_id, metadata_value)
                        .await
                        .unwrap_or_else(|_| config.to_metadata());

                    config.absorb_metadata(&stored_metadata, &overrides);
                    match apply_context_preferences(
                        &engine,
                        &active_conversation_id,
                        &mut config,
                        &overrides,
                    )
                    .await
                    {
                        Ok(restored) => {
                            // Restore database/schema context
                            if let Some(db) = restored.database {
                                if let Err(e) = engine.use_database(&db).await {
                                    println!("✗ Failed to restore database context: {}", e);
                                }
                            }
                            if let Some(schema) = restored.schema {
                                if let Err(e) = engine.use_schema(&schema).await {
                                    println!("✗ Failed to restore schema context: {}", e);
                                }
                            }
                        }
                        Err(err) => {
                            println!("✗ Failed to load conversation preferences: {}", err);
                        }
                    }
                    if let Err(err) =
                        sync_conversation_metadata(&engine, &active_conversation_id, &config).await
                    {
                        println!("✗ Failed to sync conversation metadata: {}", err);
                    }

                    let history = match refresh_conversation_history(
                        &engine,
                        &active_conversation_id,
                    )
                    .await
                    {
                        Ok(history) => history,
                        Err(err) => {
                            println!("✗ Failed to load conversation history: {}", err);
                            Vec::new()
                        }
                    };
                    agent_client.set_conversation_history(history);

                    if agent_client.get_conversation_history().is_empty() {
                        let system_prompt = build_system_prompt(&engine, &tool_executor).await;
                        agent_client.add_system_message(&system_prompt);
                    }

                    println!("🤖 Cortex Chat");
                    println!("Active conversation: {}", active_conversation_id);
                    if let Some(label) = &config.label {
                        println!("Label: {}", label);
                    }
                    println!("Model: {}", config.model);
                    print_chat_help();

                    // Initialize completer with current context
                    let mut helper = ChatHelper::new();

                    // Pre-populate completions - always fetch databases for completion
                    match engine.list_databases().await {
                        Ok(dbs) => {
                            if !dbs.is_empty() {
                                println!("📋 Loaded {} databases for tab completion", dbs.len());
                            }
                            helper.update_databases(dbs);
                        }
                        Err(e) => {
                            eprintln!("⚠ Could not load database list for completions: {}", e);
                        }
                    }
                    let current_db = engine.get_database().await.unwrap_or_default();
                    if !current_db.is_empty() {
                        if let Ok(schemas) = engine.list_schemas(Some(&current_db)).await {
                            helper.update_schemas(schemas);
                        }
                    }
                    let current_schema = engine.get_schema().await.unwrap_or_default();
                    if !current_db.is_empty() && !current_schema.is_empty() {
                        if let Ok(tables) =
                            engine.list_tables_in(&current_db, &current_schema).await
                        {
                            helper.update_tables(tables);
                        }
                    }

                    // Pre-populate models for completion
                    if let Ok(models) = engine.get_available_models().await {
                        let model_names: Vec<String> =
                            models.iter().map(|(name, _)| name.clone()).collect();
                        helper.update_models(model_names);
                    }

                    // Pre-populate warehouses for completion
                    if let Ok(warehouses) = engine.list_warehouses().await {
                        helper.update_warehouses(warehouses);
                    }

                    // Pre-populate conversations for completion
                    if let Ok(summaries) = fetch_conversation_summaries(&engine).await {
                        let conv_ids: Vec<String> =
                            summaries.iter().map(|s| s.id.clone()).collect();
                        helper.update_conversations(conv_ids);
                    }

                    // Use rustyline with custom completer for tab completion
                    let rl_config = Config::builder()
                        .completion_type(rustyline::CompletionType::List)
                        .edit_mode(rustyline::EditMode::Emacs)
                        .auto_add_history(false) // We add manually for non-empty lines
                        .build();
                    let mut rl: Editor<ChatHelper, rustyline::history::DefaultHistory> =
                        match Editor::with_config(rl_config) {
                            Ok(editor) => editor,
                            Err(e) => {
                                eprintln!("Failed to initialize line editor: {}", e);
                                return;
                            }
                        };
                    rl.set_helper(Some(helper));

                    // Try to load history from file
                    let history_file = dirs::home_dir()
                        .map(|h| h.join(".snowflake_ai_history"))
                        .unwrap_or_else(|| PathBuf::from(".snowflake_ai_history"));
                    let _ = rl.load_history(&history_file);

                    let mut last_sql_snippet: Option<String> = None;

                    loop {
                        let prompt_label =
                            config.label.as_deref().unwrap_or(&active_conversation_id);
                        let prompt = format!("You ({}): ", prompt_label);

                        let input = match rl.readline(&prompt) {
                            Ok(line) => {
                                let trimmed = line.trim().to_string();
                                if !trimmed.is_empty() {
                                    let _ = rl.add_history_entry(&trimmed);
                                }
                                trimmed
                            }
                            Err(ReadlineError::Interrupted) => {
                                println!("^C");
                                continue;
                            }
                            Err(ReadlineError::Eof) => {
                                println!("Goodbye!");
                                break;
                            }
                            Err(err) => {
                                eprintln!("Error: {:?}", err);
                                break;
                            }
                        };

                        if input.is_empty() {
                            continue;
                        }

                        if handle_document_commands(&input, &mut engine, &mut job_manager, &emitter)
                            .await
                        {
                            continue;
                        }

                        // Check for context-changing commands and update completions + save context
                        if input.starts_with("/use_database ") {
                            if handle_common_slash_commands(&input, &mut engine).await {
                                // Refresh schema completions for new database
                                let new_db = engine.get_database().await.unwrap_or_default();
                                if !new_db.is_empty() {
                                    if let Some(h) = rl.helper_mut() {
                                        if let Ok(schemas) =
                                            engine.list_schemas(Some(&new_db)).await
                                        {
                                            h.update_schemas(schemas);
                                        }
                                        h.update_tables(Vec::new()); // Clear tables until schema is selected
                                    }
                                }
                                // Save database context to conversation
                                let _ =
                                    save_database_context(&engine, &active_conversation_id).await;
                                continue;
                            }
                        }

                        if input.starts_with("/use_schema ") {
                            if handle_common_slash_commands(&input, &mut engine).await {
                                // Refresh table completions for new schema
                                let db = engine.get_database().await.unwrap_or_default();
                                let schema = engine.get_schema().await.unwrap_or_default();
                                if !db.is_empty() && !schema.is_empty() {
                                    if let Some(h) = rl.helper_mut() {
                                        if let Ok(tables) =
                                            engine.list_tables_in(&db, &schema).await
                                        {
                                            h.update_tables(tables);
                                        }
                                    }
                                }
                                // Save schema context to conversation
                                let _ =
                                    save_database_context(&engine, &active_conversation_id).await;
                                continue;
                            }
                        }

                        if handle_common_slash_commands(&input, &mut engine).await {
                            continue;
                        }

                        if input == "/exit" {
                            // Save history and database context before exit
                            let _ = save_database_context(&engine, &active_conversation_id).await;
                            let _ = rl.save_history(&history_file);
                            println!("Goodbye!");
                            break;
                        }

                        if input == "/help" {
                            print_chat_help();
                            continue;
                        }

                        if input == "/history" {
                            match refresh_conversation_history(&engine, &active_conversation_id)
                                .await
                            {
                                Ok(history) => {
                                    if history.is_empty() {
                                        println!("No conversation history.\n");
                                    } else {
                                        println!("\n📜 Conversation History:");
                                        for (idx, message) in history.iter().enumerate() {
                                            let role = message.role.as_deref().unwrap_or("unknown");
                                            println!(
                                                "{}. {}: {}",
                                                idx + 1,
                                                role,
                                                message.get_content()
                                            );
                                        }
                                        println!();
                                    }
                                }
                                Err(err) => println!("✗ Failed to fetch history: {}\n", err),
                            }
                            continue;
                        }

                        if input == "/execute" {
                            if let Some(sql) = &last_sql_snippet {
                                match engine.execute_query(sql, None, None, None).await {
                                    Ok(results) => {
                                        println!(
                                            "{}\n",
                                            serde_json::to_string_pretty(&results).unwrap_or_else(
                                                |_| "<unprintable result>".to_string()
                                            )
                                        );
                                    }
                                    Err(err) => println!("✗ Query execution failed: {}\n", err),
                                }
                            } else if let Some(sql) = prompt_for_sql_input_rl(&mut rl) {
                                match engine.execute_query(&sql, None, None, None).await {
                                    Ok(results) => println!(
                                        "{}\n",
                                        serde_json::to_string_pretty(&results)
                                            .unwrap_or_else(|_| "<unprintable result>".to_string())
                                    ),
                                    Err(err) => println!("✗ Query execution failed: {}\n", err),
                                }
                            } else {
                                println!("✗ No SQL to execute. Provide input or ask the assistant for SQL first.\n");
                            }
                            continue;
                        }

                        if let Some(sql_inline) = input.strip_prefix("/execute ") {
                            let trimmed = sql_inline.trim();
                            if trimmed.is_empty() {
                                println!("✗ Usage: /execute <sql>\n");
                            } else {
                                match engine.execute_query(trimmed, None, None, None).await {
                                    Ok(results) => println!(
                                        "{}\n",
                                        serde_json::to_string_pretty(&results)
                                            .unwrap_or_else(|_| "<unprintable result>".to_string())
                                    ),
                                    Err(err) => println!("✗ Query execution failed: {}\n", err),
                                }
                            }
                            continue;
                        }

                        if input == "/reset" {
                            match clear_conversation_messages(&engine, &active_conversation_id)
                                .await
                            {
                                Ok(_) => {
                                    agent_client.reset_conversation();
                                    let system_prompt =
                                        build_system_prompt(&engine, &tool_executor).await;
                                    agent_client.add_system_message(&system_prompt);
                                    println!("✓ Conversation history cleared.\n");
                                }
                                Err(err) => println!("✗ Failed to clear conversation: {}\n", err),
                            }
                            continue;
                        }

                        if input == "/conversations" {
                            match fetch_conversation_summaries(&engine).await {
                                Ok(summaries) => print_conversation_summaries(&summaries),
                                Err(err) => println!("✗ Failed to list conversations: {}\n", err),
                            }
                            continue;
                        }

                        if input == "/models" {
                            print_available_models(&engine).await;
                            continue;
                        }

                        if input == "/model" {
                            println!(
                                "Current model: {}\nUse /model <name> to switch.\n",
                                config.model
                            );
                            print_available_models(&engine).await;
                            continue;
                        }

                        if let Some(target) = input.strip_prefix("/use_conversation ") {
                            let trimmed = target.trim();
                            if trimmed.is_empty() {
                                println!("✗ Usage: /use_conversation <id or label>\n");
                                continue;
                            }

                            match fetch_conversation_summaries(&engine).await {
                                Ok(summaries) => {
                                    if let Some(summary) = summaries.iter().find(|summary| {
                                        summary.id == trimmed
                                            || summary
                                                .label
                                                .as_deref()
                                                .map(|label| label.eq_ignore_ascii_case(trimmed))
                                                .unwrap_or(false)
                                    }) {
                                        active_conversation_id = summary.id.clone();
                                        config.label = summary.label.clone();
                                        let metadata = engine
                                            .get_or_create_conversation(
                                                &active_conversation_id,
                                                config.to_metadata(),
                                            )
                                            .await
                                            .unwrap_or_else(|_| config.to_metadata());
                                        config.absorb_metadata(
                                            &metadata,
                                            &ConversationOverrides::default(),
                                        );
                                        match apply_context_preferences(
                                            &engine,
                                            &active_conversation_id,
                                            &mut config,
                                            &ConversationOverrides::default(),
                                        )
                                        .await
                                        {
                                            Ok(restored) => {
                                                // Restore database/schema context
                                                if let Some(db) = restored.database {
                                                    if let Err(e) = engine.use_database(&db).await {
                                                        println!("✗ Failed to restore database context: {}", e);
                                                    }
                                                }
                                                if let Some(schema) = restored.schema {
                                                    if let Err(e) = engine.use_schema(&schema).await
                                                    {
                                                        println!("✗ Failed to restore schema context: {}", e);
                                                    }
                                                }
                                                // Update completions after context change
                                                if let Some(h) = rl.helper_mut() {
                                                    let db = engine
                                                        .get_database()
                                                        .await
                                                        .unwrap_or_default();
                                                    if !db.is_empty() {
                                                        if let Ok(schemas) =
                                                            engine.list_schemas(Some(&db)).await
                                                        {
                                                            h.update_schemas(schemas);
                                                        }
                                                    }
                                                    let schema = engine
                                                        .get_schema()
                                                        .await
                                                        .unwrap_or_default();
                                                    if !db.is_empty() && !schema.is_empty() {
                                                        if let Ok(tables) = engine
                                                            .list_tables_in(&db, &schema)
                                                            .await
                                                        {
                                                            h.update_tables(tables);
                                                        }
                                                    }
                                                }
                                            }
                                            Err(err) => {
                                                println!(
                                                    "✗ Failed to load conversation preferences: {}",
                                                    err
                                                );
                                            }
                                        }
                                        if let Err(err) = sync_conversation_metadata(
                                            &engine,
                                            &active_conversation_id,
                                            &config,
                                        )
                                        .await
                                        {
                                            println!(
                                                "✗ Failed to sync conversation metadata: {}",
                                                err
                                            );
                                        }

                                        match refresh_conversation_history(
                                            &engine,
                                            &active_conversation_id,
                                        )
                                        .await
                                        {
                                            Ok(history) => {
                                                agent_client.set_conversation_history(history)
                                            }
                                            Err(err) => {
                                                println!(
                                                    "✗ Failed to load conversation history: {}",
                                                    err
                                                );
                                                agent_client.reset_conversation();
                                            }
                                        }

                                        if agent_client.get_conversation_history().is_empty() {
                                            let system_prompt =
                                                build_system_prompt(&engine, &tool_executor).await;
                                            agent_client.add_system_message(&system_prompt);
                                        }

                                        println!(
                                            "✓ Switched to conversation {}",
                                            active_conversation_id
                                        );
                                    } else {
                                        println!("✗ Conversation not found: {}\n", trimmed);
                                    }
                                }
                                Err(err) => println!("✗ Failed to list conversations: {}\n", err),
                            }
                            continue;
                        }

                        if let Some(label_value) = input.strip_prefix("/label ") {
                            let trimmed = label_value.trim();
                            if trimmed.is_empty() {
                                config.label = None;
                                println!("✓ Cleared conversation label.\n");
                            } else {
                                config.label = Some(trimmed.to_string());
                                println!("✓ Updated conversation label to '{}'.\n", trimmed);
                            }
                            if let Err(err) = sync_conversation_metadata(
                                &engine,
                                &active_conversation_id,
                                &config,
                            )
                            .await
                            {
                                println!("✗ Failed to update metadata: {}", err);
                            }
                            continue;
                        }

                        if let Some(model_value) = input.strip_prefix("/model ") {
                            let trimmed = model_value.trim();
                            if trimmed.is_empty() {
                                println!("✗ Usage: /model <name>\n");
                            } else {
                                config.model = trimmed.to_string();
                                println!("✓ Model set to {}\n", trimmed);
                                if let Err(err) = sync_conversation_metadata(
                                    &engine,
                                    &active_conversation_id,
                                    &config,
                                )
                                .await
                                {
                                    println!("✗ Failed to update metadata: {}", err);
                                }
                            }
                            continue;
                        }

                        if let Some(temp_value) = input.strip_prefix("/temperature ") {
                            let trimmed = temp_value.trim();
                            match trimmed.parse::<f32>() {
                                Ok(value) if (0.0..=1.0).contains(&value) => {
                                    config.temperature = Some(value);
                                    println!("✓ Temperature set to {}\n", value);
                                    if let Err(err) = sync_conversation_metadata(
                                        &engine,
                                        &active_conversation_id,
                                        &config,
                                    )
                                    .await
                                    {
                                        println!("✗ Failed to update metadata: {}", err);
                                    }
                                }
                                _ => {
                                    println!("✗ Temperature must be a number between 0.0 and 1.0\n")
                                }
                            }
                            continue;
                        }

                        if let Some(token_value) = input.strip_prefix("/max_tokens ") {
                            let trimmed = token_value.trim();
                            match trimmed.parse::<i32>() {
                                Ok(value) if value > 0 => {
                                    config.max_tokens = Some(value);
                                    println!("✓ Max tokens set to {}\n", value);
                                    if let Err(err) = sync_conversation_metadata(
                                        &engine,
                                        &active_conversation_id,
                                        &config,
                                    )
                                    .await
                                    {
                                        println!("✗ Failed to update metadata: {}", err);
                                    }
                                }
                                _ => println!("✗ max_tokens must be a positive integer\n"),
                            }
                            continue;
                        }

                        // /text2sql - Generate SQL from natural language
                        if let Some(prompt) = input.strip_prefix("/text2sql ") {
                            let prompt = prompt.trim();
                            if prompt.is_empty() {
                                println!("✗ Usage: /text2sql <natural language prompt>\n");
                            } else {
                                match engine.execute_with_cortex_context(prompt, false).await {
                                    Ok(result) => {
                                        let sql_str = if let Some(s) = result.as_str() {
                                            s.to_string()
                                        } else {
                                            result.to_string()
                                        };
                                        println!("\n📝 Generated SQL:\n```sql\n{}\n```\n", sql_str);
                                        last_sql_snippet = Some(sql_str);
                                    }
                                    Err(e) => println!("✗ Failed to generate SQL: {}\n", e),
                                }
                            }
                            continue;
                        }

                        // /context - Show database context
                        if input == "/context" {
                            let db = current_database_name(&engine).await;
                            let schema = current_schema_name(&engine).await;
                            println!("\n📊 Database Context:");
                            println!("  Database: {}", db);
                            println!("  Schema: {}", schema);

                            // Count tables
                            match engine.list_tables_in(&db, &schema).await {
                                Ok(tables) => println!("  Tables: {}", tables.len()),
                                Err(_) => println!("  Tables: (unable to count)"),
                            }
                            println!();
                            continue;
                        }

                        // /stats - Show conversation statistics
                        if input == "/stats" {
                            let usage_stats = agent_client.get_usage_stats();
                            let history_len = agent_client.get_conversation_history().len();

                            println!("\n📈 Conversation Statistics:");
                            println!("  Conversation ID: {}", active_conversation_id);
                            if let Some(label) = &config.label {
                                println!("  Label: {}", label);
                            }
                            println!("  Model: {}", config.model);
                            println!("  Messages: {}", history_len);
                            println!("  Total Requests: {}", usage_stats.total_requests);
                            println!("  Prompt Tokens: {}", usage_stats.total_prompt_tokens);
                            println!(
                                "  Completion Tokens: {}",
                                usage_stats.total_completion_tokens
                            );
                            println!("  Total Tokens: {}", usage_stats.total_tokens);
                            println!();
                            continue;
                        }

                        // /jobs - List document processing jobs
                        if input == "/jobs" || input.starts_with("/jobs ") {
                            let args = input.strip_prefix("/jobs").unwrap_or("").trim();
                            if let Err(e) = jobs::handle_jobs_command(&engine, args, &emitter).await
                            {
                                println!("✗ {}\n", e);
                            }
                            continue;
                        }

                        // /cancel - Cancel a document processing job
                        if let Some(job_id) = input.strip_prefix("/cancel ") {
                            if let Err(e) = jobs::handle_cancel_command(
                                &engine,
                                &mut job_manager,
                                job_id,
                                &emitter,
                            )
                            .await
                            {
                                println!("✗ {}\n", e);
                            }
                            continue;
                        }

                        // /timeout - Get or set operation timeout
                        if input == "/timeout" || input.starts_with("/timeout ") {
                            let args = input.strip_prefix("/timeout").unwrap_or("").trim();
                            if let Err(e) = jobs::handle_timeout_command(&mut job_manager, args) {
                                println!("✗ {}\n", e);
                            }
                            continue;
                        }

                        // /tools - List available AI tools
                        if input == "/tools" {
                            println!("\n🔧 Available Tools:");
                            for tool_name in tool_executor.registry().list_tools() {
                                if let Some(tool) = tool_executor.registry().get_tool(tool_name) {
                                    println!("  {} - {}", tool.name, tool.description);
                                }
                            }
                            println!();
                            continue;
                        }

                        if input.starts_with('/') {
                            println!("✗ Unrecognized command: {}", input);
                            println!("Type /help to see the list of available commands.\n");
                            continue;
                        }

                        let user_message = input.clone();
                        let mut request_messages = agent_client.get_conversation_history();
                        request_messages.push(agents::Message::new_user(user_message.clone()));
                        agent_client.set_conversation_history(request_messages.clone());
                        if let Err(err) = persist_conversation_message(
                            &engine,
                            &active_conversation_id,
                            "user",
                            &user_message,
                        )
                        .await
                        {
                            println!("✗ Failed to record user message: {}", err);
                        }

                        if !emitter.is_json_mode() {
                            print!("\n🤖 Assistant: ");
                            std::io::stdout().flush().unwrap();
                        }

                        // Run the agentic loop with tool calling support
                        // We need to wrap engine temporarily for the agentic loop
                        let engine_arc = std::sync::Arc::new(tokio::sync::Mutex::new(engine));

                        let agentic_result = run_agentic_loop(
                            engine_arc.clone(),
                            &mut agent_client,
                            &tool_executor,
                            &emitter,
                            &mut job_manager,
                            &config.model,
                            config.temperature,
                            config.max_tokens,
                            request_messages,
                            None, // No cancellation token for now
                        )
                        .await;

                        // Restore engine from Arc<Mutex>
                        // This is safe because we're the only owner at this point
                        engine = match std::sync::Arc::try_unwrap(engine_arc) {
                            Ok(mutex) => mutex.into_inner(),
                            Err(_) => {
                                panic!("Engine Arc should have single owner after agentic loop");
                            }
                        };

                        match agentic_result {
                            Ok(result) => {
                                if !emitter.is_json_mode() {
                                    println!("\n");
                                }

                                if !result.cancelled && !result.response_text.is_empty() {
                                    agent_client
                                        .add_assistant_response(result.response_text.clone());

                                    // Persist message in background - don't block the UI
                                    // We clone the necessary data and spawn the persistence
                                    let response_text = result.response_text.clone();
                                    let conv_id = active_conversation_id.clone();

                                    // Fire and forget - persist asynchronously
                                    // Errors will be silently ignored but that's OK for UX
                                    tokio::spawn({
                                        let engine_clone = engine.clone();
                                        let config_clone = config.clone();
                                        async move {
                                            let _ = persist_conversation_message(
                                                &engine_clone,
                                                &conv_id,
                                                "assistant",
                                                &response_text,
                                            )
                                            .await;
                                            let _ = sync_conversation_metadata(
                                                &engine_clone,
                                                &conv_id,
                                                &config_clone,
                                            )
                                            .await;
                                        }
                                    });

                                    // Extract SQL snippets from response
                                    let snippets = extract_sql_snippets(&result.response_text);
                                    if let Some(snippet) = snippets.last() {
                                        last_sql_snippet = Some(snippet.clone());
                                    }
                                }
                            }
                            Err(err) => {
                                if emitter.is_json_mode() {
                                    emitter.error(&err, Some("AGENTIC_ERROR"));
                                } else {
                                    println!("\n✗ Error: {}\n", err);
                                }
                            }
                        }
                    }
                }
                Err(e) => println!("Failed to connect: {}", e),
            }
        }
    }
}
