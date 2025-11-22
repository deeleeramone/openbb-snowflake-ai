mod agents;
mod cache;
mod engine;
mod lsp;

use crate::engine::SnowflakeEngine;

use chrono::TimeZone;
use clap::{Parser, Subcommand};
use futures::StreamExt;
use serde_json::json;
use std::path::{Path, PathBuf};
use tokio::io::{self, AsyncBufReadExt};

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
        #[arg(short = 'e', long)]
        env_file: Option<PathBuf>,
    },
    /// Interactive AI_COMPLETE session with configurable parameters
    AiComplete {
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

        #[arg(short = 'm', long, default_value = "llama3.1-70b")]
        model: String,

        #[arg(short = 't', long)]
        temperature: Option<f32>,

        #[arg(long)]
        max_tokens: Option<i32>,

        #[arg(short = 'c', long)]
        context_file: Option<PathBuf>,

        #[arg(short = 'e', long)]
        env_file: Option<PathBuf>,
    },
    /// Interactive AI agent with streaming responses
    Agent {
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

        #[arg(short = 'm', long, default_value = "llama3.1-70b")]
        model: String,

        #[arg(short = 't', long)]
        temperature: Option<f32>,

        #[arg(long)]
        max_tokens: Option<i32>,

        #[arg(short = 'e', long)]
        env_file: Option<PathBuf>,

        #[arg(long, default_value = "true")]
        stream: bool,
    },
    /// List all accessible databases
    ListDatabases {
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
        #[arg(short = 'e', long)]
        env_file: Option<PathBuf>,
    },
    /// List all schemas in a database
    ListSchemas {
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
        #[arg(long)]
        target_database: Option<String>,
        #[arg(short = 'e', long)]
        env_file: Option<PathBuf>,
    },
    /// List all accessible warehouses
    ListWarehouses {
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
        #[arg(short = 'e', long)]
        env_file: Option<PathBuf>,
    },
    /// List all tables across all accessible databases and schemas
    ListAllTables {
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
        #[arg(short = 'e', long)]
        env_file: Option<PathBuf>,
    },
}

fn print_chat_help() {
    println!("Commands:");
    println!("  /help                     - Show this command menu");
    println!("  /reset                    - Clear conversation history");
    println!("  /history                  - Show conversation history");
    println!("  /execute                  - Execute the last generated SQL query");
    println!("  /stage                    - Set semantic model stage");
    println!("  /model                    - Set semantic model file");
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
    println!("  /use_database <db_name>   - Switch to a different database");
    println!("  /use_schema <schema_name> - Switch to a different schema");
    println!("  /use_warehouse <wh_name>  - Switch to a different warehouse");
    println!("  /current                  - Show current database/schema");
    println!("  /exit                     - Exit chat");
    println!();
}

fn print_ai_complete_help() {
    println!("Commands:");
    println!("  /help                     - Show this command menu");
    println!("  /execute                  - Execute a SQL query");
    println!("  /models                   - List available models");
    println!("  /model <name>             - Change AI model");
    println!("  /temperature <val>        - Set temperature (0.0-1.0)");
    println!("  /max_tokens <val>         - Set max tokens");
    println!("  /context <file>           - Load context from file");
    println!("  /clear_context            - Clear loaded context");
    println!("  /show_context             - Show current context");
    println!("  /history                  - Show conversation history");
    println!("  /reset                    - Clear conversation history");
    println!("  /settings                 - Show current settings");
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
    println!("  /use_database <db_name>   - Switch to a different database");
    println!("  /use_schema <schema_name> - Switch to a different schema");
    println!("  /use_warehouse <wh_name>  - Switch to a different warehouse");
    println!("  /current                  - Show current database/schema");
    println!("  /exit                     - Exit session");
    println!();
}

fn print_agent_help() {
    println!("Commands:");
    println!("  /help                     - Show this command menu");
    println!("  /execute                  - Execute a SQL query");
    println!("  /model <name>             - Change AI model");
    println!("  /models                   - List available models");
    println!("  /temperature <val>        - Set temperature (0.0-1.0)");
    println!("  /max_tokens <val>         - Set max tokens");
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
    println!("  /use_database <db_name>   - Switch to a different database");
    println!("  /use_schema <schema_name> - Switch to a different schema");
    println!("  /use_warehouse <wh_name>  - Switch to a different warehouse");
    println!("  /current                  - Show current database/schema");
    println!("  /agents                   - List all agents");
    println!("  /sessions                 - List sessions for an agent");
    println!("  /create_agent             - Create a new agent");
    println!("  /create_agent_from_file   - Create agent from JSON file");
    println!("  /create_session           - Create a new session");
    println!("  /get_agent <id>           - Get agent by ID");
    println!("  /get_session <id>         - Get session by ID");
    println!("  /update_agent <id>        - Update an agent");
    println!("  /delete_agent <id>        - Delete an agent");
    println!("  /delete_session <id>      - Delete a session");
    println!("  /send_to_session          - Send message to a session");
    println!("  /list_messages <id>       - List messages in a session");
    println!("  /get_message              - Get a specific message");
    println!("  /usage                    - Show API usage statistics");
    println!("  /reset                    - Clear conversation history");
    println!("  /exit                     - Exit session");
    println!("  /tools                    - Use function calling");
    println!("  /system                   - Add system message");
    println!();
}

async fn handle_common_slash_commands(input: &str, engine: &mut SnowflakeEngine) -> bool {
    if input == "/current" {
        println!("\n📍 Current Context:");
        println!("  Account: {}", engine.get_current_account());
        println!("  Database: {}", engine.get_current_database());
        println!("  Schema: {}", engine.get_current_schema());
        println!();
        return true;
    }

    if input == "/databases" {
        match engine.list_databases().await {
            Ok(databases) => {
                println!("\n📚 Accessible Databases:");
                for db in databases {
                    let marker = if db == engine.get_current_database() {
                        " (current)"
                    } else {
                        ""
                    };
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

        match engine.list_schemas(target_db).await {
            Ok(schemas) => {
                let db_name = target_db.unwrap_or(engine.get_current_database());
                println!("\n📂 Schemas in {}:", db_name);
                for s in schemas {
                    let marker = if s == engine.get_current_schema() && target_db.is_none() {
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
        match engine.get_table_list().await {
            Ok(tables) => {
                println!(
                    "\n📋 Tables in {}.{}:",
                    engine.get_current_database(),
                    engine.get_current_schema()
                );
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
            match engine.get_table_info(table_name).await {
                Ok(columns) => {
                    println!(
                        "Table: {}.{}.{}",
                        engine.get_current_database(),
                        engine.get_current_schema(),
                        table_name
                    );
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

                    println!(
                        "Table: {}.{}.{}",
                        engine.get_current_database(),
                        engine.get_current_schema(),
                        table_name
                    );
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
            // Use ai_complete directly without expensive context
            match engine
                .ai_complete_raw(
                    "llama3.1-70b",
                    &format!("Complete this SQL query: {}", partial_query),
                    None,
                    None,
                )
                .await
            {
                Ok(completion) => {
                    println!("\n🤖 Completed Query:\n```sql\n{}\n```", completion);
                }
                Err(e) => println!("✗ Failed to complete query: {}\n", e),
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
        let default_path_str = "/Users/darrenlee/.openbb_platform/.env";
        let default_path = Path::new(default_path_str);
        if default_path.exists() {
            dotenvy::from_path(default_path).expect("Failed to load default .env file");
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
            ..
        } => {
            match SnowflakeEngine::new(
                &user, &password, &account, &role, &warehouse, &database, &schema,
            )
            .await
            {
                Ok(mut engine) => {
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

                    if !engine.has_semantic_model_stage() {
                        println!("✗ No semantic model configured.");
                        println!("Please either:");
                        println!(
                            "  1. Set SNOWFLAKE_FILE and SNOWFLAKE_STAGE environment variables"
                        );
                        println!("  2. Use the -m/--semantic-model flag to specify a semantic model file");
                        return;
                    }

                    if !engine.is_cortex_enabled() {
                        println!("✗ Cortex is not enabled. Check your SNOWFLAKE_PASSWORD (PAT).");
                        return;
                    }

                    println!("🤖 Cortex Analyst Interactive Chat");
                    print_chat_help();
                    let mut lines = io::BufReader::new(io::stdin()).lines();
                    let mut last_sql: Option<String> = None;

                    loop {
                        print!("You: ");
                        use std::io::Write;
                        std::io::stdout().flush().unwrap();

                        if let Ok(Some(input)) = lines.next_line().await {
                            let input = input.trim();

                            if input.is_empty() {
                                continue;
                            }

                            if handle_common_slash_commands(input, &mut engine).await {
                                continue;
                            }

                            match input {
                                "/exit" => {
                                    println!("Goodbye!");
                                    break;
                                }
                                "/help" => {
                                    print_chat_help();
                                    continue;
                                }
                                "/reset" => {
                                    engine.reset_conversation();
                                    last_sql = None;
                                    println!("✓ Conversation history cleared.\n");
                                    continue;
                                }
                                "/history" => {
                                    let history = engine.get_conversation_history();
                                    if history.is_empty() {
                                        println!("No conversation history.\n");
                                    } else {
                                        println!("\n📜 Conversation History:");
                                        for (i, msg) in history.iter().enumerate() {
                                            println!("{}. {}: {:?}", i + 1, msg.role, msg.content);
                                        }
                                        println!();
                                    }
                                    continue;
                                }
                                "/execute" => {
                                    if let Some(sql) = &last_sql {
                                        println!("\n⚡ Executing SQL...\n");

                                        let cleaned_sql = sql
                                            .lines()
                                            .filter(|line| !line.trim().starts_with("--"))
                                            .collect::<Vec<_>>()
                                            .join("\n")
                                            .trim()
                                            .trim_end_matches(';')
                                            .to_string();

                                        match engine
                                            .execute_query(&cleaned_sql, None, None, None)
                                            .await
                                        {
                                            Ok(results) => {
                                                println!(
                                                    "{}\n",
                                                    serde_json::to_string_pretty(&results).unwrap()
                                                );
                                            }
                                            Err(e) => println!("✗ Query execution failed: {}\n", e),
                                        }
                                    } else {
                                        println!(
                                            "✗ No SQL query to execute. Ask a question first.\n"
                                        );
                                    }
                                    continue;
                                }
                                "/stage" => {
                                    let (current_stage, current_file) =
                                        engine.get_semantic_model_config();

                                    let stage_name = if let Some(stage_path) = current_stage {
                                        stage_path
                                            .trim_start_matches('@')
                                            .split('/')
                                            .next()
                                            .and_then(|s| s.split('.').nth(2))
                                            .unwrap_or("")
                                    } else {
                                        ""
                                    };

                                    let file_name = if let Some(file) = current_file {
                                        file
                                    } else if let Some(stage_path) = current_stage {
                                        stage_path.split('/').nth(1).unwrap_or("")
                                    } else {
                                        ""
                                    };

                                    println!();
                                    if !stage_name.is_empty() {
                                        println!("Current stage: {}", stage_name);
                                    }
                                    if !file_name.is_empty() {
                                        println!("Current model: {}", file_name);
                                    }

                                    match engine.list_stages().await {
                                        Ok(stages) if !stages.is_empty() => {
                                            println!("\nAvailable stages:");
                                            for stage in &stages {
                                                println!("  - {}", stage);
                                            }
                                        }
                                        _ => {}
                                    }

                                    println!();
                                    print!("Enter stage name (e.g., SEMANTIC_MODELS): ");
                                    std::io::stdout().flush().unwrap();

                                    if let Ok(Some(stage_input)) = lines.next_line().await {
                                        let stage_input = stage_input.trim();

                                        if stage_input.is_empty() {
                                            println!("Cancelled.\n");
                                            continue;
                                        }

                                        let file = current_file.unwrap_or("semantic_model.yaml");
                                        engine.switch_semantic_model(
                                            stage_input.to_string(),
                                            file.to_string(),
                                        );
                                        engine.reset_conversation();
                                        last_sql = None;

                                        println!("✓ Stage set to: {}", stage_input);
                                        println!("✓ Conversation history cleared.\n");
                                    }
                                    continue;
                                }
                                _ => {}
                            }

                            if input.starts_with("/stage ") {
                                let (current_stage, current_file) =
                                    engine.get_semantic_model_config();

                                let stage_name = if let Some(stage_path) = current_stage {
                                    stage_path
                                        .trim_start_matches('@')
                                        .split('/')
                                        .next()
                                        .and_then(|s| s.split('.').nth(2))
                                        .unwrap_or("")
                                } else {
                                    ""
                                };

                                let file_name = if let Some(file) = current_file {
                                    file
                                } else if let Some(stage_path) = current_stage {
                                    stage_path.split('/').nth(1).unwrap_or("")
                                } else {
                                    ""
                                };

                                println!();
                                if !stage_name.is_empty() {
                                    println!("Current stage: {}", stage_name);
                                }
                                if !file_name.is_empty() {
                                    println!("Current model: {}", file_name);
                                }

                                if !stage_name.is_empty() {
                                    if let Ok(files) = engine.list_files_in_stage(stage_name).await {
                                        let yaml_files: Vec<_> = files
                                            .iter()
                                            .filter(|f| {
                                                f.ends_with(".yaml") || f.ends_with(".yml")
                                            })
                                            .collect();

                                        if !yaml_files.is_empty() {
                                            println!("\nAvailable models in {}:", stage_name);
                                            for file in &yaml_files {
                                                println!("  - {}", file);
                                            }
                                        }
                                    }
                                }

                                println!();
                                print!("Enter file name (e.g., cybersyn.yaml): ");
                                std::io::stdout().flush().unwrap();

                                if let Ok(Some(file_input)) = lines.next_line().await {
                                    let file_input = file_input.trim();

                                    if file_input.is_empty() {
                                        println!("Cancelled.\n");
                                        continue;
                                    }

                                    let stage = if !stage_name.is_empty() {
                                        stage_name.to_string()
                                    } else {
                                        "SEMANTIC_MODELS".to_string()
                                    };

                                    engine.switch_semantic_model(stage, file_input.to_string());
                                    engine.reset_conversation();
                                    last_sql = None;

                                    println!("✓ Model set to: {}", file_input);
                                    println!("✓ Conversation history cleared.\n");
                                }
                                continue;
                            }

                            if input.starts_with('/') {
                                println!("✗ Unrecognized command: {}", input);
                                println!("Type /help to see the list of available commands.\n");
                                continue;
                            }

                            match engine.chat_with_analyst(input, true).await {
                                Ok(response) => {
                                    println!("\n🤖 Assistant:");

                                    let mut found_sql = false;
                                    for content in &response.message.content {
                                        match content {
                                            engine::CortexResponseContent::Text { text } => {
                                                println!("{}", text);
                                            }
                                            engine::CortexResponseContent::Sql { statement } => {
                                                println!("\n📝 SQL:\n```sql\n{}\n```", statement);
                                                last_sql = Some(statement.clone());
                                                found_sql = true;
                                            }
                                            engine::CortexResponseContent::Suggestions {
                                                suggestions,
                                            } => {
                                                println!("\n💡 Suggestions:");
                                                for (i, suggestion) in
                                                    suggestions.iter().enumerate()
                                                {
                                                    println!("  {}. {}", i + 1, suggestion);
                                                }
                                            }
                                        }
                                    }

                                    if found_sql {
                                        println!("\n💡 Tip: Type /execute to run this query");
                                    }
                                    println!();
                                }
                                Err(e) => println!("✗ Error: {}\n", e),
                            }
                        }
                    }
                }
                Err(e) => println!("Failed to connect: {}", e),
            }
        }
        Commands::AiComplete {
            user,
            password,
            account,
            role,
            warehouse,
            database,
            schema,
            model,
            temperature,
            max_tokens,
            context_file,
            ..
        } => {
            match SnowflakeEngine::new(
                &user, &password, &account, &role, &warehouse, &database, &schema,
            )
            .await
            {
                Ok(mut engine) => {
                    let mut current_model = model;
                    let mut current_temperature = temperature;
                    let mut current_max_tokens = max_tokens;
                    let mut context_text = String::new();

                    // Load context file if provided
                    if let Some(context_path) = context_file {
                        match std::fs::read_to_string(&context_path) {
                            Ok(content) => {
                                context_text = content;
                                println!("✓ Loaded context from: {}", context_path.display());
                            }
                            Err(e) => {
                                println!("✗ Failed to load context file: {}", e);
                                return;
                            }
                        }
                    }

                    println!("🤖 AI_COMPLETE Interactive Session");
                    println!("Current settings:");
                    println!("  Model: {}", current_model);
                    if let Some(temp) = current_temperature {
                        println!("  Temperature: {}", temp);
                    } else {
                        println!("  Temperature: 0.7 (default)");
                    }
                    if let Some(max) = current_max_tokens {
                        println!("  Max tokens: {}", max);
                    } else {
                        println!("  Max tokens: 4096 (default)");
                    }
                    print_ai_complete_help();
                    let mut lines = io::BufReader::new(io::stdin()).lines();
                    let mut conversation_history: Vec<(String, String)> = Vec::new();

                    loop {
                        print!("Prompt: ");
                        use std::io::Write;
                        std::io::stdout().flush().unwrap();

                        if let Ok(Some(input)) = lines.next_line().await {
                            let input = input.trim();

                            if input.is_empty() {
                                continue;
                            }

                            if handle_common_slash_commands(input, &mut engine).await {
                                continue;
                            }

                            if input == "/exit" {
                                println!("Goodbye!");
                                break;
                            }

                            if input == "/help" {
                                print_ai_complete_help();
                                continue;
                            }

                            if input == "/reset" {
                                conversation_history.clear();
                                println!("✓ Conversation history cleared\n");
                                continue;
                            }

                            if input == "/execute" {
                                println!(
                                    "Enter SQL query to execute (end with ';' and an empty line):"
                                );
                                let mut query_buffer = String::new();
                                loop {
                                    match lines.next_line().await {
                                        Ok(Some(line)) => {
                                            if line.is_empty() && query_buffer.trim().ends_with(';')
                                            {
                                                let query =
                                                    query_buffer.trim().trim_end_matches(';');
                                                if !query.trim().is_empty() {
                                                    println!("\n⚡ Executing SQL...\n");
                                                    match engine
                                                        .execute_query(query, None, None, None)
                                                        .await
                                                    {
                                                        Ok(results) => {
                                                            println!(
                                                                "{}\n",
                                                                serde_json::to_string_pretty(
                                                                    &results
                                                                )
                                                                .unwrap()
                                                            );
                                                        }
                                                        Err(e) => println!(
                                                            "✗ Query execution failed: {}\n",
                                                            e
                                                        ),
                                                    }
                                                }
                                                break;
                                            }
                                            query_buffer.push_str(&line);
                                            query_buffer.push('\n');
                                        }
                                        Err(_) | Ok(None) => {
                                            println!("\n✗ Input stream closed. Aborting.");
                                            break;
                                        }
                                    }
                                }
                                print!("Prompt: ");
                                std::io::stdout().flush().unwrap();
                                continue;
                            }

                            if input == "/history" {
                                if conversation_history.is_empty() {
                                    println!("✗ No conversation history\n");
                                } else {
                                    println!("\n📜 Conversation History:");
                                    for (i, (user_msg, ai_msg)) in
                                        conversation_history.iter().enumerate()
                                    {
                                        println!("\n{}. You: {}", i + 1, user_msg);
                                        println!("   AI: {}", ai_msg);
                                    }
                                    println!();
                                }
                                continue;
                            }

                            if input == "/models" {
                                println!("\n📋 Querying available models...\n");
                                match engine.get_available_models().await {
                                    Ok(models) => {
                                        println!("Available models:");
                                        for (name, desc) in models {
                                            println!("  • {}: {}", name, desc);
                                        }
                                    }
                                    Err(e) => {
                                        println!("✗ Failed to fetch models: {}", e);
                                    }
                                }
                                println!();
                                continue;
                            }

                            if input == "/settings" {
                                println!("\nCurrent settings:");
                                println!("  Model: {}", current_model);
                                if let Some(temp) = current_temperature {
                                    println!("  Temperature: {}", temp);
                                } else {
                                    println!("  Temperature: 0.7 (default)");
                                }
                                if let Some(max) = current_max_tokens {
                                    println!("  Max tokens: {}", max);
                                } else {
                                    println!("  Max tokens: 4096 (default)");
                                }
                                if !context_text.is_empty() {
                                    println!("  Context: {} characters loaded", context_text.len());
                                } else {
                                    println!("  Context: none");
                                }
                                println!();
                                continue;
                            }

                            if input == "/show_context" {
                                if context_text.is_empty() {
                                    println!("✗ No context loaded\n");
                                } else {
                                    println!(
                                        "\n📄 Current Context ({} chars):",
                                        context_text.len()
                                    );
                                    println!("{}", context_text);
                                    println!();
                                }
                                continue;
                            }

                            if input == "/clear_context" {
                                context_text.clear();
                                println!("✓ Context cleared\n");
                                continue;
                            }

                            // Handle slash commands with arguments - AiComplete mode
                            if input.starts_with("/context ") {
                                let file_path = input.strip_prefix("/context ").unwrap().trim();
                                match std::fs::read_to_string(file_path) {
                                    Ok(content) => {
                                        context_text = content;
                                        println!(
                                            "✓ Loaded context from: {} ({} chars)\n",
                                            file_path,
                                            context_text.len()
                                        );
                                    }
                                    Err(e) => {
                                        println!("✗ Failed to load context file: {}", e);
                                    }
                                }
                                continue;
                            }

                            if input.starts_with("/model ") {
                                let new_model = input.strip_prefix("/model ").unwrap().trim();
                                if !new_model.is_empty() {
                                    current_model = new_model.to_string();
                                    println!("✓ Model set to: {}\n", current_model);
                                } else {
                                    println!("✗ Invalid model name\n");
                                }
                                continue;
                            }

                            if input.starts_with("/temperature ") {
                                let temp_str = input.strip_prefix("/temperature ").unwrap().trim();
                                match temp_str.parse::<f32>() {
                                    Ok(temp) if (0.0..=1.0).contains(&temp) => {
                                        current_temperature = Some(temp);
                                        println!("✓ Temperature set to: {}\n", temp);
                                    }
                                    _ => println!("✗ Invalid temperature (must be 0.0-1.0)\n"),
                                }
                                continue;
                            }

                            if input.starts_with("/max_tokens ") {
                                let tokens_str = input.strip_prefix("/max_tokens ").unwrap().trim();
                                match tokens_str.parse::<i32>() {
                                    Ok(tokens) if tokens > 0 => {
                                        current_max_tokens = Some(tokens);
                                        println!("✓ Max tokens set to: {}\n", tokens);
                                    }
                                    _ => println!(
                                        "✗ Invalid max_tokens (must be positive integer)\n"
                                    ),
                                }
                                continue;
                            }

                            if input.starts_with('/') {
                                println!("✗ Unrecognized command: {}", input);
                                println!("Type /help to see the list of available commands.\n");
                                continue;
                            }

                            // Build conversation context from history
                            let mut full_prompt = String::new();

                            // Add document context if available
                            if !context_text.is_empty() {
                                full_prompt.push_str("DOCUMENT CONTEXT:\n");
                                full_prompt.push_str(&context_text);
                                full_prompt.push_str("\n\n");
                            }

                            // Add conversation history
                            if !conversation_history.is_empty() {
                                full_prompt.push_str("CONVERSATION HISTORY:\n");
                                for (user_msg, ai_msg) in &conversation_history {
                                    full_prompt.push_str(&format!(
                                        "User: {}\nAssistant: {}\n\n",
                                        user_msg, ai_msg
                                    ));
                                }
                            }

                            // Add current question
                            if !context_text.is_empty() || !conversation_history.is_empty() {
                                full_prompt.push_str("CURRENT QUESTION:\n");
                            }
                            full_prompt.push_str(input);

                            // Build options with higher defaults for long context
                            let options = {
                                let mut opts = serde_json::Map::new();

                                // Use higher defaults when context is loaded
                                let default_max_tokens =
                                    if !context_text.is_empty() { 4096 } else { 2048 };

                                opts.insert(
                                    "temperature".to_string(),
                                    json!(current_temperature.unwrap_or(0.7)),
                                );
                                opts.insert(
                                    "max_tokens".to_string(),
                                    json!(current_max_tokens.unwrap_or(default_max_tokens)),
                                );

                                Some(serde_json::Value::Object(opts))
                            };

                            match engine
                                .ai_complete(&current_model, &full_prompt, options, None)
                                .await
                            {
                                Ok(response) => {
                                    println!("\n🤖 Response:\n{}\n", response);

                                    // Add to conversation history
                                    conversation_history.push((input.to_string(), response));

                                    // Keep only last 10 exchanges to avoid context overflow
                                    if conversation_history.len() > 10 {
                                        conversation_history.remove(0);
                                    }
                                }
                                Err(e) => println!("✗ AI_COMPLETE failed: {}\n", e),
                            }
                        }
                    }
                }
                Err(e) => println!("Failed to connect: {}", e),
            }
        }
        Commands::Agent {
            user,
            password,
            account,
            role,
            warehouse,
            database,
            schema,
            model,
            temperature,
            max_tokens,
            stream,
            ..
        } => {
            match SnowflakeEngine::new(
                &user, &password, &account, &role, &warehouse, &database, &schema,
            )
            .await
            {
                Ok(mut engine) => {
                    let mut agent_client = match engine.create_agents_client() {
                        Ok(client) => client,
                        Err(e) => {
                            println!("✗ Failed to create agents client: {}", e);
                            return;
                        }
                    };

                    let mut current_model = model;
                    let mut current_temperature = temperature;
                    let mut current_max_tokens = max_tokens;
                    let use_streaming = stream;

                    println!("🤖 Cortex Agent Interactive Session");
                    println!("Current settings:");
                    println!("  Model: {}", current_model);
                    println!(
                        "  Streaming: {}",
                        if use_streaming { "enabled" } else { "disabled" }
                    );
                    if let Some(temp) = current_temperature {
                        println!("  Temperature: {}", temp);
                    }
                    if let Some(max) = current_max_tokens {
                        println!("  Max tokens: {}", max);
                    }
                    print_agent_help();
                    let mut lines = io::BufReader::new(io::stdin()).lines();

                    loop {
                        print!("You: ");
                        use std::io::Write;
                        std::io::stdout().flush().unwrap();

                        if let Ok(Some(input)) = lines.next_line().await {
                            let input = input.trim();

                            if input.is_empty() {
                                continue;
                            }

                            if handle_common_slash_commands(input, &mut engine).await {
                                continue;
                            }

                            if input == "/exit" {
                                println!("Goodbye!");
                                break;
                            }

                            if input == "/help" {
                                print_agent_help();
                                continue;
                            }

                            if input == "/reset" {
                                agent_client.reset_conversation();
                                println!("✓ Conversation history cleared\n");
                                continue;
                            }

                            if input == "/execute" {
                                println!(
                                    "Enter SQL query to execute (end with ';' and an empty line):"
                                );
                                let mut query_buffer = String::new();
                                loop {
                                    match lines.next_line().await {
                                        Ok(Some(line)) => {
                                            if line.is_empty() && query_buffer.trim().ends_with(';')
                                            {
                                                let query =
                                                    query_buffer.trim().trim_end_matches(';');
                                                if !query.trim().is_empty() {
                                                    println!("\n⚡ Executing SQL...\n");
                                                    match engine
                                                        .execute_query(query, None, None, None)
                                                        .await
                                                    {
                                                        Ok(results) => {
                                                            println!(
                                                                "{}\n",
                                                                serde_json::to_string_pretty(
                                                                    &results
                                                                )
                                                                .unwrap()
                                                            );
                                                        }
                                                        Err(e) => println!(
                                                            "✗ Query execution failed: {}\n",
                                                            e
                                                        ),
                                                    }
                                                }
                                                break;
                                            }
                                            query_buffer.push_str(&line);
                                            query_buffer.push('\n');
                                        }
                                        Err(_) | Ok(None) => {
                                            println!("\n✗ Input stream closed. Aborting.");
                                            break;
                                        }
                                    }
                                }
                                print!("You: ");
                                std::io::stdout().flush().unwrap();
                                continue;
                            }

                            if input == "/history" {
                                let history = agent_client.get_conversation_history();
                                if history.is_empty() {
                                    println!("✗ No conversation history\n");
                                } else {
                                    println!("\n📜 Conversation History:");
                                    for (i, msg) in history.iter().enumerate() {
                                        let role = msg
                                            .role.as_deref()
                                            .unwrap_or("unknown");
                                        println!("\n{}. {}: {}", i + 1, role, msg.get_content());
                                    }
                                    println!();
                                }
                                continue;
                            }

                            if input == "/models" {
                                println!("\n📋 Querying available models...\n");
                                match engine.get_available_models().await {
                                    Ok(models) => {
                                        println!("Available models:");
                                        for (name, desc) in models {
                                            println!("  • {}: {}", name, desc);
                                        }
                                    }
                                    Err(e) => {
                                        println!("✗ Failed to fetch models: {}", e);
                                    }
                                }
                                println!();
                                continue;
                            }

                            if input.starts_with("/model ") {
                                let new_model = input.strip_prefix("/model ").unwrap().trim();
                                if !new_model.is_empty() {
                                    current_model = new_model.to_string();
                                    println!("✓ Model set to: {}\n", current_model);
                                } else {
                                    println!("✗ Invalid model name\n");
                                }
                                continue;
                            }

                            if input.starts_with("/temperature ") {
                                let temp_str = input.strip_prefix("/temperature ").unwrap().trim();
                                match temp_str.parse::<f32>() {
                                    Ok(temp) if (0.0..=1.0).contains(&temp) => {
                                        current_temperature = Some(temp);
                                        println!("✓ Temperature set to: {}\n", temp);
                                    }
                                    _ => println!("✗ Invalid temperature (must be 0.0-1.0)\n"),
                                }
                                continue;
                            }

                            if input.starts_with("/max_tokens ") {
                                let tokens_str = input.strip_prefix("/max_tokens ").unwrap().trim();
                                match tokens_str.parse::<i32>() {
                                    Ok(tokens) if tokens > 0 => {
                                        current_max_tokens = Some(tokens);
                                        println!("✓ Max tokens set to: {}\n", tokens);
                                    }
                                    _ => println!(
                                        "✗ Invalid max_tokens (must be positive integer)\n"
                                    ),
                                }
                                continue;
                            }

                            if input == "/agents" {
                                match agent_client.list_agents(None, None, None).await {
                                    Ok(agents) => {
                                        println!("\n👥 Available Agents:");
                                        if agents.is_empty() {
                                            println!("  No agents found.");
                                        } else {
                                            for agent in agents {
                                                println!(
                                                    "  - ID: {}",
                                                    agent.id.unwrap_or_default()
                                                );
                                                println!("    Name: {}", agent.name);
                                                println!(
                                                    "    Description: {}",
                                                    agent
                                                        .comment
                                                        .unwrap_or_else(|| "N/A".to_string())
                                                );
                                                println!("    Model: {}", agent.models.model);
                                                println!(
                                                    "    Created: {}",
                                                    agent
                                                        .created_on
                                                        .map(|ts| chrono::Local
                                                            .timestamp_opt(ts, 0)
                                                            .unwrap()
                                                            .to_string())
                                                        .unwrap_or_else(|| "N/A".to_string())
                                                );
                                                println!();
                                            }
                                        }
                                    }
                                    Err(e) => println!("✗ Failed to list agents: {}\n", e),
                                }
                                continue;
                            }

                            if input == "/sessions" {
                                print!("Enter agent ID: ");
                                std::io::stdout().flush().unwrap();
                                if let Ok(Some(agent_id_input)) = lines.next_line().await {
                                    let agent_id = agent_id_input.trim();
                                    if agent_id.is_empty() {
                                        println!("✗ Agent ID cannot be empty.\n");
                                        continue;
                                    }

                                    match agent_client.list_sessions(agent_id).await {
                                        Ok(sessions) => {
                                            println!("\n💬 Sessions:");
                                            if sessions.is_empty() {
                                                println!(
                                                    "  No sessions found for agent {}.",
                                                    agent_id
                                                );
                                            } else {
                                                for session in sessions {
                                                    println!(
                                                        "  - ID: {}",
                                                        session.id.unwrap_or_default()
                                                    );
                                                    println!("    Agent ID: {}", session.agent_id);
                                                    println!(
                                                        "    Created: {}",
                                                        session
                                                            .created_on
                                                            .map(|ts| chrono::Local
                                                                .timestamp_opt(ts, 0)
                                                                .unwrap()
                                                                .to_string())
                                                            .unwrap_or_else(|| "N/A".to_string())
                                                    );
                                                    println!();
                                                }
                                            }
                                        }
                                        Err(e) => println!("✗ Failed to list sessions: {}\n", e),
                                    }
                                }
                                continue;
                            }

                            if input.starts_with("/get_agent ") {
                                let agent_id = input.strip_prefix("/get_agent ").unwrap().trim();
                                if !agent_id.is_empty() {
                                    match agent_client.get_agent(agent_id).await {
                                        Ok(agent) => {
                                            println!("\n👥 Agent Details:");
                                            println!("  ID: {}", agent.id.unwrap_or_default());
                                            println!("  Name: {}", agent.name);
                                            println!(
                                                "  Description: {}",
                                                agent.comment.unwrap_or_else(|| "N/A".to_string())
                                            );
                                            println!("  Model: {}", agent.models.model);
                                            println!(
                                                "  Created: {}",
                                                agent
                                                    .created_on
                                                    .map(|ts| chrono::Local
                                                        .timestamp_opt(ts, 0)
                                                        .unwrap()
                                                        .to_string())
                                                    .unwrap_or_else(|| "N/A".to_string())
                                            );
                                            println!();
                                        }
                                        Err(e) => println!("✗ Failed to get agent: {}\n", e),
                                    }
                                } else {
                                    println!("✗ Usage: /get_agent <id>\n");
                                }
                                continue;
                            }

                            if input.starts_with("/delete_agent ") {
                                let agent_id = input.strip_prefix("/delete_agent ").unwrap().trim();
                                if !agent_id.is_empty() {
                                    match agent_client.delete_agent(agent_id, None).await {
                                        Ok(_) => println!("✓ Agent {} deleted.\n", agent_id),
                                        Err(e) => println!("✗ Failed to delete agent: {}\n", e),
                                    }
                                } else {
                                    println!("✗ Usage: /delete_agent <id>\n");
                                }
                                continue;
                            }

                            if input == "/create_agent" {
                                println!("\n➕ Creating new agent:");
                                print!("  Agent Name: ");
                                std::io::stdout().flush().unwrap();
                                let name = if let Ok(Some(n)) = lines.next_line().await {
                                    n.trim().to_string()
                                } else {
                                    "".to_string()
                                };

                                print!("  Agent Description (optional): ");
                                std::io::stdout().flush().unwrap();
                                let description = if let Ok(Some(d)) = lines.next_line().await {
                                    let d_trim = d.trim();
                                    if d_trim.is_empty() {
                                        None
                                    } else {
                                        Some(d_trim.to_string())
                                    }
                                } else {
                                    None
                                };

                                print!("  Agent Model (default: {}): ", current_model);
                                std::io::stdout().flush().unwrap();
                                let model = if let Ok(Some(m)) = lines.next_line().await {
                                    let m_trim = m.trim();
                                    if m_trim.is_empty() {
                                        current_model.clone()
                                    } else {
                                        m_trim.to_string()
                                    }
                                } else {
                                    current_model.clone()
                                };

                                if name.is_empty() {
                                    println!("✗ Agent name cannot be empty. Cancelled.\n");
                                    continue;
                                }

                                let model_config = agents::ModelConfig {
                                    model: model.clone(),
                                    temperature: current_temperature,
                                    top_p: None,
                                    max_tokens: current_max_tokens,
                                };

                                let instructions = agents::AgentInstructions {
                                    response: "You are a helpful assistant.".to_string(), // Default instruction
                                    orchestration: None,
                                    system: None,
                                    sample_questions: None,
                                };

                                let create_req = agents::CreateAgentRequest {
                                    name: name.clone(),
                                    comment: description,
                                    profile: None, // No profile input for now
                                    models: model_config,
                                    instructions,
                                    orchestration: None,
                                    tools: None,
                                    tool_resources: None,
                                };

                                match agent_client.create_agent(create_req, None).await {
                                    Ok(agent) => println!(
                                        "✓ Agent created with ID: {}\n",
                                        agent.id.unwrap_or_default()
                                    ),
                                    Err(e) => println!("✗ Failed to create agent: {}\n", e),
                                }
                                continue;
                            }

                            if input.starts_with("/create_agent_from_file ") {
                                let file_path = input
                                    .strip_prefix("/create_agent_from_file ")
                                    .unwrap()
                                    .trim();
                                if !file_path.is_empty() {
                                    match agent_client.create_agent_from_file(file_path, None).await
                                    {
                                        Ok(agent) => println!(
                                            "✓ Agent created from file with ID: {}\n",
                                            agent.id.unwrap_or_default()
                                        ),
                                        Err(e) => {
                                            println!("✗ Failed to create agent from file: {}\n", e)
                                        }
                                    }
                                } else {
                                    println!("✗ Usage: /create_agent_from_file <file_path>\n");
                                }
                                continue;
                            }

                            if input.starts_with("/update_agent ") {
                                let parts: Vec<&str> = input.splitn(2, ' ').collect();
                                if parts.len() == 2 {
                                    let agent_id = parts[1].trim();
                                    println!("\n✏️ Updating agent {}:", agent_id);

                                    print!("  New Name (press Enter to keep current): ");
                                    std::io::stdout().flush().unwrap();
                                    let name = if let Ok(Some(n)) = lines.next_line().await {
                                        let n_trim = n.trim();
                                        if n_trim.is_empty() {
                                            None
                                        } else {
                                            Some(n_trim.to_string())
                                        }
                                    } else {
                                        None
                                    };

                                    print!("  New Description (press Enter to keep current): ");
                                    std::io::stdout().flush().unwrap();
                                    let description = if let Ok(Some(d)) = lines.next_line().await {
                                        let d_trim = d.trim();
                                        if d_trim.is_empty() {
                                            None
                                        } else {
                                            Some(d_trim.to_string())
                                        }
                                    } else {
                                        None
                                    };

                                    print!("  New Model (press Enter to keep current): ");
                                    std::io::stdout().flush().unwrap();
                                    let model = if let Ok(Some(m)) = lines.next_line().await {
                                        let m_trim = m.trim();
                                        if m_trim.is_empty() {
                                            None
                                        } else {
                                            Some(m_trim.to_string())
                                        }
                                    } else {
                                        None
                                    };

                                    let mut update_req = agents::UpdateAgentRequest {
                                        name,
                                        comment: description,
                                        profile: None,
                                        models: None,
                                        instructions: None,
                                        orchestration: None,
                                        tools: None,
                                        tool_resources: None,
                                    };

                                    if let Some(m) = model {
                                        update_req.models = Some(agents::ModelConfig {
                                            model: m,
                                            temperature: current_temperature,
                                            top_p: None,
                                            max_tokens: current_max_tokens,
                                        });
                                    }

                                    match agent_client.update_agent(agent_id, update_req).await {
                                        Ok(agent) => println!(
                                            "✓ Agent {} updated.\n",
                                            agent.id.unwrap_or_default()
                                        ),
                                        Err(e) => println!("✗ Failed to update agent: {}\n", e),
                                    }
                                } else {
                                    println!("✗ Usage: /update_agent <id>\n");
                                }
                                continue;
                            }

                            if input == "/sessions" {
                                print!("Enter agent ID (press Enter for default agent): ");
                                std::io::stdout().flush().unwrap();
                                if let Ok(Some(agent_id_input)) = lines.next_line().await {
                                    let agent_id = if agent_id_input.trim().is_empty() {
                                        None
                                    } else {
                                        Some(agent_id_input.trim().to_string())
                                    };

                                    match agent_client.list_sessions(&agent_id.unwrap()).await {
                                        Ok(sessions) => {
                                            println!("\n💬 Sessions:");
                                            if sessions.is_empty() {
                                                println!("  No sessions found.");
                                            } else {
                                                for session in sessions {
                                                    println!("  - ID: {}", session.id.unwrap());
                                                    println!("    Agent ID: {}", session.agent_id);
                                                    println!(
                                                        "    Created: {}",
                                                        session.created_on.unwrap()
                                                    );
                                                    println!();
                                                }
                                            }
                                        }
                                        Err(e) => println!("✗ Failed to list sessions: {}\n", e),
                                    }
                                }
                                continue;
                            }

                            if input.starts_with("/get_session ") {
                                let session_id =
                                    input.strip_prefix("/get_session ").unwrap().trim();
                                if !session_id.is_empty() {
                                    match agent_client.get_session(session_id).await {
                                        Ok(session) => {
                                            println!("\n💬 Session Details:");
                                            println!("  ID: {}", session.id.unwrap().trim());
                                            println!("  Agent ID: {}", session.agent_id);
                                            println!("  Created: {}", session.created_on.unwrap());
                                            println!();
                                        }
                                        Err(e) => println!("✗ Failed to get session: {}\n", e),
                                    }
                                } else {
                                    println!("✗ Usage: /get_session <id>\n");
                                }
                                continue;
                            }

                            if input.starts_with("/delete_session ") {
                                let session_id =
                                    input.strip_prefix("/delete_session ").unwrap().trim();
                                if !session_id.is_empty() {
                                    match agent_client.delete_session(session_id).await {
                                        Ok(_) => println!("✓ Session {} deleted.\n", session_id),
                                        Err(e) => println!("✗ Failed to delete session: {}\n", e),
                                    }
                                } else {
                                    println!("✗ Usage: /delete_session <id>\n");
                                }
                                continue;
                            }

                            if input == "/create_session" {
                                println!("\n➕ Creating new session:");
                                print!("  Agent ID: ");
                                std::io::stdout().flush().unwrap();
                                let agent_id_input = if let Ok(Some(id)) = lines.next_line().await {
                                    id.trim().to_string()
                                } else {
                                    "".to_string()
                                };

                                if agent_id_input.is_empty() {
                                    println!("✗ Agent ID cannot be empty. Cancelled.\n");
                                    continue;
                                }

                                match agent_client.create_session(&agent_id_input, None).await {
                                    Ok(session) => println!(
                                        "✓ Session created with ID: {}\n",
                                        session.id.unwrap_or_default()
                                    ),
                                    Err(e) => println!("✗ Failed to create session: {}\n", e),
                                }
                                continue;
                            }

                            if input.starts_with("/send_to_session ") {
                                let parts: Vec<&str> = input.splitn(2, ' ').collect();
                                if parts.len() == 2 {
                                    let session_id = parts[1].trim();
                                    print!("Message: ");
                                    std::io::stdout().flush().unwrap();
                                    if let Ok(Some(message)) = lines.next_line().await {
                                        let message_trim = message.trim();
                                        if !message_trim.is_empty() {
                                            match agent_client
                                                .send_message(session_id, message_trim)
                                                .await
                                            {
                                                Ok(_) => println!(
                                                    "✓ Message sent to session {}.\n",
                                                    session_id
                                                ),
                                                Err(e) => {
                                                    println!("✗ Failed to send message: {}\n", e)
                                                }
                                            }
                                        } else {
                                            println!("✗ Message cannot be empty. Cancelled.\n");
                                        }
                                    }
                                } else {
                                    println!("✗ Usage: /send_to_session <session_id>\n");
                                }
                                continue;
                            }

                            if input.starts_with("/list_messages ") {
                                let session_id =
                                    input.strip_prefix("/list_messages ").unwrap().trim();
                                if !session_id.is_empty() {
                                    match agent_client.list_messages(session_id).await {
                                        Ok(messages) => {
                                            println!("\n✉️ Messages in session {}:", session_id);
                                            if messages.is_empty() {
                                                println!("  No messages found.");
                                            } else {
                                                for msg in messages {
                                                    let role = msg.role;
                                                    println!("  - {}: {:?}", role, msg.content);
                                                }
                                            }
                                            println!();
                                        }
                                        Err(e) => println!("✗ Failed to list messages: {}\n", e),
                                    }
                                } else {
                                    println!("✗ Usage: /list_messages <session_id>\n");
                                }
                                continue;
                            }

                            if input.starts_with("/get_message ") {
                                let parts: Vec<&str> = input
                                    .strip_prefix("/get_message ")
                                    .unwrap()
                                    .splitn(1, ' ')
                                    .collect();
                                if parts.len() == 1 {
                                    let session_id = parts[0].trim();
                                    print!("Message ID: ");
                                    std::io::stdout().flush().unwrap();
                                    let message_id = if let Ok(Some(id)) = lines.next_line().await {
                                        id.trim().to_string()
                                    } else {
                                        "".to_string()
                                    };

                                    if message_id.is_empty() {
                                        println!("✗ Message ID cannot be empty. Cancelled.\n");
                                        continue;
                                    }

                                    match agent_client.get_message(session_id, &message_id).await {
                                        Ok(msg) => {
                                            println!("\n✉️ Message Details:");
                                            println!("  ID: {}", msg.id.unwrap_or_default());
                                            println!("  Session ID: {}", msg.session_id);
                                            println!("  Role: {}", msg.role);
                                            println!("  Content: {:?}", msg.content);
                                            println!(
                                                "  Created: {}",
                                                msg.created_on
                                                    .map(|ts| chrono::Local
                                                        .timestamp_opt(ts, 0)
                                                        .unwrap()
                                                        .to_string())
                                                    .unwrap_or_else(|| "N/A".to_string())
                                            );
                                            println!();
                                        }
                                        Err(e) => println!("✗ Failed to get message: {}\n", e),
                                    }
                                } else {
                                    println!("✗ Usage: /get_message <session_id>\n");
                                }
                                continue;
                            }

                            if input == "/usage" {
                                println!(
                                    "\n📊 API Usage Statistics:\n{}",
                                    agent_client.get_usage_report()
                                );
                                continue;
                            }

                            if input == "/tools" {
                                println!("\n🔧 Function Calling Tools: (Not yet implemented)\n");
                                continue;
                            }

                            if input.starts_with("/system ") {
                                let system_message = input.strip_prefix("/system ").unwrap().trim();
                                if !system_message.is_empty() {
                                    agent_client.add_system_message(system_message);
                                    println!("✓ System message added.\n");
                                } else {
                                    println!("✗ Usage: /system <message>\n");
                                }
                                continue;
                            }

                            if input == "/current" {
                                println!("\n📍 Current Context:");
                                println!("  Account: {}", engine.get_current_account());
                                println!("  Database: {}", engine.get_current_database());
                                println!("  Schema: {}", engine.get_current_schema());
                                println!();
                                continue;
                            }

                            if input == "/databases" {
                                match engine.list_databases().await {
                                    Ok(databases) => {
                                        println!("\n📚 Accessible Databases:");
                                        for db in databases {
                                            let marker = if db == engine.get_current_database() {
                                                " (current)"
                                            } else {
                                                ""
                                            };
                                            println!("  - {}{}", db, marker);
                                        }
                                        println!();
                                    }
                                    Err(e) => println!("✗ Failed to list databases: {}\n", e),
                                }
                                continue;
                            }

                            if input == "/schemas" {
                                print!("Database name (press Enter for current): ");
                                std::io::stdout().flush().unwrap();
                                if let Ok(Some(db_input)) = lines.next_line().await {
                                    let db = if db_input.trim().is_empty() {
                                        None
                                    } else {
                                        Some(db_input.trim())
                                    };
                                    match engine.list_schemas(db).await {
                                        Ok(schemas) => {
                                            let db_name =
                                                db.unwrap_or(engine.get_current_database());
                                            println!("\n📂 Schemas in {}:", db_name);
                                            for s in schemas {
                                                let marker = if s == engine.get_current_schema()
                                                    && db.is_none()
                                                {
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
                                }
                                continue;
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
                                continue;
                            }

                            if input == "/tables" {
                                match engine.get_table_list().await {
                                    Ok(tables) => {
                                        println!(
                                            "\n📋 Tables in {}.{}:",
                                            engine.get_current_database(),
                                            engine.get_current_schema()
                                        );
                                        for table in tables {
                                            println!("  - {}", table);
                                        }
                                        println!();
                                    }
                                    Err(e) => println!("✗ Failed to list tables: {}\n", e),
                                }
                                continue;
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
                                continue;
                            }

                            if input.starts_with("/use_database ") {
                                let db_name = input.strip_prefix("/use_database ").unwrap().trim();
                                if !db_name.is_empty() {
                                    match engine.use_database(db_name).await {
                                        Ok(_) => println!("✓ Switched to database: {}\n", db_name),
                                        Err(e) => println!("✗ Failed to switch database: {}\n", e),
                                    }
                                }
                                continue;
                            }

                            if input.starts_with("/use_warehouse ") {
                                let wh_name = input.strip_prefix("/use_warehouse ").unwrap().trim();
                                if !wh_name.is_empty() {
                                    match engine.use_warehouse(wh_name).await {
                                        Ok(_) => println!("✓ Switched to warehouse: {}\n", wh_name),
                                        Err(e) => println!("✗ Failed to switch warehouse: {}\n", e),
                                    }
                                }
                                continue;
                            }

                            if input.starts_with("/use_schema ") {
                                let schema_name =
                                    input.strip_prefix("/use_schema ").unwrap().trim();
                                if !schema_name.is_empty() {
                                    match engine.use_schema(schema_name).await {
                                        Ok(_) => {
                                            println!("✓ Switched to schema: {}\n", schema_name)
                                        }
                                        Err(e) => println!("✗ Failed to switch schema: {}\n", e),
                                    }
                                }
                                continue;
                            }

                            if input.starts_with('/') {
                                println!("✗ Unrecognized command: {}", input);
                                println!("Type /help to see the list of available commands.\n");
                                continue;
                            }
                            // Process user message
                            if use_streaming {
                                print!("\n🤖 Assistant: ");
                                std::io::stdout().flush().unwrap();

                                match agent_client
                                    .stream_complete(
                                        &current_model,
                                        vec![agents::Message::new_user(input.to_string())],
                                        current_temperature,
                                        None, // top_p
                                        current_max_tokens,
                                        None, // tools
                                    )
                                    .await
                                {
                                    Ok((mut stream, mut metadata)) => {
                                        let mut full_response = String::new();
                                        let mut last_chunk_had_usage = false;

                                        while let Some(chunk_result) = stream.next().await {
                                            match chunk_result {
                                                Ok(chunk) => {
                                                    // Update metadata with actual values from chunk
                                                    if metadata.request_id.is_empty()
                                                        && !chunk.id.is_empty()
                                                    {
                                                        metadata.request_id = chunk.id.clone();
                                                        metadata.model = chunk.model.clone();
                                                    }

                                                    // Capture usage from last chunk
                                                    if let Some(usage) = chunk.usage {
                                                        metadata.usage = Some(usage);
                                                        last_chunk_had_usage = true;
                                                    }

                                                    for choice in chunk.choices {
                                                        // The role is not streamed in the delta, it's part of the initial message.
                                                        // If we need to display the role, it should be handled before the streaming loop
                                                        // or derived from the overall conversation context.
                                                        // For now, remove the attempt to access choice.delta.role.

                                                        if let Some(content) = &choice.delta.text {
                                                            // Changed from .content to .text
                                                            print!("{}", content);
                                                            std::io::stdout().flush().unwrap();
                                                            full_response.push_str(content);
                                                        }

                                                        // Check for finish reason
                                                        if let Some(finish_reason) =
                                                            &choice.finish_reason
                                                        {
                                                            println!(
                                                                "\n[Finished: {}]",
                                                                finish_reason
                                                            );
                                                        }
                                                    }
                                                }
                                                Err(e) => {
                                                    println!("\n✗ Stream error: {}", e);
                                                    break;
                                                }
                                            }
                                        }

                                        // Add assistant response to history after streaming completes
                                        if !full_response.is_empty() {
                                            agent_client.add_assistant_response(full_response);
                                        }

                                        // Update usage stats if we got usage data
                                        if last_chunk_had_usage {
                                            agent_client.update_metadata_and_stats(metadata);
                                        }

                                        println!("\n");
                                    }
                                    Err(e) => println!("\n✗ Error: {}\n", e),
                                }
                            } else {
                                match agent_client
                                    .complete(
                                        &current_model,
                                        input,
                                        true,
                                        current_temperature,
                                        None,
                                        current_max_tokens,
                                    )
                                    .await
                                {
                                    Ok(response) => {
                                        if let Some(choice) = response.choices.first() {
                                            let content = choice.message.get_content();

                                            println!("\n🤖 Assistant: {}\n", content);

                                            // Display tool calls if present
                                            if let Some(tool_calls) = &choice.tool_calls {
                                                println!("🔧 Tool Calls:");
                                                for call in tool_calls {
                                                    println!("  ID: {}", call.id);
                                                    println!("  Type: {}", call.tool_type);
                                                    println!("  Function: {}", call.function.name);
                                                    println!(
                                                        "  Arguments: {}",
                                                        call.function.arguments
                                                    );
                                                }
                                                println!();
                                            }

                                            // Get and display metadata
                                            if let Some(metadata) = agent_client.get_last_metadata()
                                            {
                                                println!("ℹ️  Metadata:");
                                                println!("   ID: {}", metadata.request_id);
                                                println!("   Model: {}", metadata.model);
                                                println!("   Created: {}", metadata.created);
                                                println!("   Object: {}", metadata.object);
                                                println!("   Choice Index: {}", choice.index);
                                                if let Some(finish_reason) = &choice.finish_reason {
                                                    println!("   Finish Reason: {}", finish_reason);
                                                }
                                                if let Some(usage) = &metadata.usage {
                                                    println!("   Tokens: {} prompt + {} completion = {} total",
                                                        usage.prompt_tokens, usage.completion_tokens, usage.total_tokens);
                                                }
                                            }
                                            println!();
                                        }
                                    }
                                    Err(e) => println!("\n✗ Error: {}\n", e),
                                }
                            }
                        }
                    }
                }
                Err(e) => println!("Failed to connect: {}", e),
            }
        }
        Commands::ListDatabases {
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
                Ok(engine) => match engine.list_databases().await {
                    Ok(databases) => {
                        println!("Accessible databases:");
                        for db in databases {
                            println!("  - {}", db);
                        }
                    }
                    Err(e) => println!("Failed to list databases: {}", e),
                },
                Err(e) => println!("Failed to connect: {}", e),
            }
        }
        Commands::ListSchemas {
            user,
            password,
            account,
            role,
            warehouse,
            database,
            schema,
            target_database,
            ..
        } => {
            match SnowflakeEngine::new(
                &user, &password, &account, &role, &warehouse, &database, &schema,
            )
            .await
            {
                Ok(engine) => match engine.list_schemas(target_database.as_deref()).await {
                    Ok(schemas) => {
                        let db_name = target_database.as_ref().unwrap_or(&database);
                        println!("Schemas in {}:", db_name);
                        for s in schemas {
                            println!("  - {}", s);
                        }
                    }
                    Err(e) => println!("Failed to list schemas: {}", e),
                },
                Err(e) => println!("Failed to connect: {}", e),
            }
        }
        Commands::ListWarehouses {
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
                Ok(engine) => match engine.list_warehouses().await {
                    Ok(warehouses) => {
                        println!("Accessible warehouses:");
                        for wh in warehouses {
                            println!("  - {}", wh);
                        }
                    }
                    Err(e) => println!("Failed to list warehouses: {}", e),
                },
                Err(e) => println!("Failed to connect: {}", e),
            }
        }
        Commands::ListAllTables {
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
                    println!("Fetching all accessible tables... (this may take a moment)");
                    match engine.list_all_tables().await {
                        Ok(tables) => {
                            println!("\nAll accessible tables:");
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
                        }
                        Err(e) => println!("Failed to list all tables: {}", e),
                    }
                }
                Err(e) => println!("Failed to connect: {}", e),
            }
        }
    }

    // After all commands have potentially run and the engine has been used,
    // attempt to close the cache connection if it was initialized.
    if let Some(cache_arc_mutex) = engine::CACHE_DB.get() {
        let mut cache_guard = cache_arc_mutex.lock().await;
        if let Err(e) = cache_guard.close().await {
            eprintln!("Error closing cache connection: {}", e);
        }
    }
}
