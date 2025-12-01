mod agents;
mod engine;
mod lsp;

use crate::engine::SnowflakeEngine;

use clap::{Parser, Subcommand};
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tokio::io::{self, AsyncBufReadExt};
use uuid::Uuid;

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

async fn apply_context_preferences(
    engine: &SnowflakeEngine,
    conversation_id: &str,
    config: &mut ConversationConfig,
    overrides: &ConversationOverrides,
) -> Result<(), String> {
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

    Ok(())
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

    match config.label.as_ref().map(|label| label.trim()).filter(|s| !s.is_empty()) {
        Some(label) => {
            set_context_value(engine, conversation_id, LABEL_PREFERENCE_KEY, label).await?;
        }
        None => {
            let _ = delete_context_value(engine, conversation_id, LABEL_PREFERENCE_KEY).await;
        }
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

async fn handle_document_commands(input: &str, engine: &mut SnowflakeEngine) -> bool {
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
            .unwrap_or("upload.bin");
        match fs::read(&path) {
            Ok(bytes) => match engine.upload_bytes_to_stage(&bytes, file_name, None).await {
                Ok(stage_path) => {
                    println!("✓ Uploaded {} to {}\n", file_name, stage_path);
                }
                Err(err) => println!("✗ Upload failed: {}\n", err),
            },
            Err(err) => println!("✗ Failed to read file: {}\n", err),
        }
        return true;
    }

    if let Some(stage_path) = input.strip_prefix("/parse ") {
        let stage_path = stage_path.trim();
        if stage_path.is_empty() {
            println!("✗ Usage: /parse <stage_path>\n");
            return true;
        }
        match engine.ai_parse_document(stage_path, Some("LAYOUT")).await {
            Ok(result) => {
                println!(
                    "✓ Parsed document. Result preview (first 500 chars):\n{}\n",
                    &result.chars().take(500).collect::<String>()
                );
            }
            Err(err) => println!("✗ Document parsing failed: {}\n", err),
        }
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

async fn build_system_prompt(engine: &SnowflakeEngine) -> String {
    let database = engine
        .get_database()
        .await
        .unwrap_or_else(|_| "UNKNOWN".to_string());
    let schema = engine
        .get_schema()
        .await
        .unwrap_or_else(|_| "UNKNOWN".to_string());
    let user_schema = engine.user_schema_name();

    format!(
        "You are the OpenBB Cortex Analyst acting against the Snowflake environment. \
Database context: {database}.{schema}. \
Conversations, context objects, and attachments must be stored in OPENBB_AGENTS.{user_schema}.AGENTS_CONVERSATIONS, \
OPENBB_AGENTS.{user_schema}.AGENTS_MESSAGES, and OPENBB_AGENTS.{user_schema}.AGENTS_CONTEXT_OBJECTS. \
Only reference documents available via DOCUMENT_PARSE_RESULTS, CORTEX_UPLOADS, or AGENTS_CONTEXT_OBJECTS. \
Always provide concise, accurate answers grounded in Snowflake data."
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
            ..
        } => {
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
                    if let Err(err) = apply_context_preferences(
                        &engine,
                        &active_conversation_id,
                        &mut config,
                        &overrides,
                    )
                    .await
                    {
                        println!("✗ Failed to load conversation preferences: {}", err);
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
                        let system_prompt = build_system_prompt(&engine).await;
                        agent_client.add_system_message(&system_prompt);
                    }

                    println!("🤖 Cortex Chat");
                    println!("Active conversation: {}", active_conversation_id);
                    if let Some(label) = &config.label {
                        println!("Label: {}", label);
                    }
                    println!("Model: {}", config.model);
                    print_chat_help();

                    let mut lines = io::BufReader::new(io::stdin()).lines();
                    let mut last_sql_snippet: Option<String> = None;

                    loop {
                        let prompt_label =
                            config.label.as_deref().unwrap_or(&active_conversation_id);
                        print!("You ({}): ", prompt_label);
                        std::io::stdout().flush().unwrap();

                        let input = match lines.next_line().await {
                            Ok(Some(line)) => line.trim().to_string(),
                            _ => break,
                        };

                        if input.is_empty() {
                            continue;
                        }

                        if handle_document_commands(&input, &mut engine).await {
                            continue;
                        }

                        if handle_common_slash_commands(&input, &mut engine).await {
                            continue;
                        }

                        if input == "/exit" {
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
                            } else if let Some(sql) = prompt_for_sql_input(&mut lines).await {
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
                                    let system_prompt = build_system_prompt(&engine).await;
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
                                        if let Err(err) = apply_context_preferences(
                                            &engine,
                                            &active_conversation_id,
                                            &mut config,
                                            &ConversationOverrides::default(),
                                        )
                                        .await
                                        {
                                            println!(
                                                "✗ Failed to load conversation preferences: {}",
                                                err
                                            );
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
                                            let system_prompt = build_system_prompt(&engine).await;
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

                        print!("\n🤖 Assistant: ");
                        std::io::stdout().flush().unwrap();

                        match agent_client
                            .stream_complete(
                                &config.model,
                                request_messages,
                                config.temperature,
                                None,
                                config.max_tokens,
                                None,
                            )
                            .await
                        {
                            Ok((mut stream, metadata)) => {
                                let mut full_response = String::new();
                                while let Some(chunk_result) = stream.next().await {
                                    match chunk_result {
                                        Ok(chunk) => {
                                            for choice in chunk.choices {
                                                if let Some(content) = &choice.delta.text {
                                                    print!("{}", content);
                                                    std::io::stdout().flush().unwrap();
                                                    full_response.push_str(content);
                                                }
                                            }
                                        }
                                        Err(err) => {
                                            println!("\n✗ Stream error: {}", err);
                                            break;
                                        }
                                    }
                                }
                                println!("\n");
                                agent_client.update_metadata_and_stats(metadata);
                                if !full_response.is_empty() {
                                    agent_client.add_assistant_response(full_response.clone());
                                    if let Err(err) = persist_conversation_message(
                                        &engine,
                                        &active_conversation_id,
                                        "assistant",
                                        &full_response,
                                    )
                                    .await
                                    {
                                        println!("✗ Failed to record assistant message: {}", err);
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
                                    if let Some(snippet) = extract_last_sql_snippet(&full_response)
                                    {
                                        last_sql_snippet = Some(snippet);
                                    }
                                }
                            }
                            Err(err) => println!("\n✗ Error: {}\n", err),
                        }
                    }
                }
                Err(e) => println!("Failed to connect: {}", e),
            }
        }
    }
}
