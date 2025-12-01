use crate::engine::{Schema, SnowflakeEngine};
use dashmap::DashMap;
use regex::Regex;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer, LspService, Server};

#[derive(Debug)]
struct Backend {
    client: Client,
    engine: Arc<Mutex<Option<SnowflakeEngine>>>,
    document_map: Arc<DashMap<String, String>>,
    schema_cache: Arc<Mutex<Vec<Schema>>>,
}

#[derive(Debug, Clone, PartialEq)]
enum CompletionContext {
    TableName,
    ColumnName,
    FunctionName,
    Keyword,
    DataType,
    WindowFunction,
    JoinCondition,
    CaseExpression,
    SubQuery,
}

struct QueryPosition {
    context: CompletionContext,
    partial_word: String,
    table_qualifier: Option<String>,
}

impl Backend {
    fn get_word_at_position(&self, text: &str, position: Position) -> (String, Range) {
        let lines: Vec<&str> = text.lines().collect();
        if position.line as usize >= lines.len() {
            return (String::new(), Range::new(position, position));
        }

        let line = lines[position.line as usize];
        let char_pos = position.character as usize;

        if char_pos > line.len() {
            return (String::new(), Range::new(position, position));
        }

        let mut start = char_pos;
        while start > 0 {
            let ch = line.chars().nth(start - 1).unwrap_or(' ');
            if ch.is_alphanumeric() || ch == '_' {
                start -= 1;
            } else {
                break;
            }
        }

        let word = line[start..char_pos].to_string();
        let range = Range::new(
            Position::new(position.line, start as u32),
            Position::new(position.line, char_pos as u32),
        );

        (word, range)
    }

    fn analyze_query_context(&self, text: &str, position: Position) -> QueryPosition {
        let lines: Vec<&str> = text.lines().collect();
        let char_pos = position.character as usize;

        let mut text_before_cursor = String::new();
        for (i, line) in lines.iter().enumerate() {
            if i < position.line as usize {
                text_before_cursor.push_str(line);
                text_before_cursor.push(' ');
            } else if i == position.line as usize {
                let end = char_pos.min(line.len());
                text_before_cursor.push_str(&line[..end]);
                break;
            }
        }

        let (partial_word, _) = self.get_word_at_position(text, position);
        let upper_text = text_before_cursor.to_uppercase();

        // SubQuery context - inside parentheses after SELECT/IN/EXISTS
        if let Some(open_paren_pos) = text_before_cursor.rfind('(') {
            let text_after_paren = &text_before_cursor[open_paren_pos..];
            if Regex::new(r"(?i)^\(\s*(SELECT|WITH)")
                .unwrap()
                .is_match(text_after_paren)
            {
                return QueryPosition {
                    context: CompletionContext::SubQuery,
                    partial_word,
                    table_qualifier: None,
                };
            }
        }

        // Qualified column (table.column or alias.column)
        if let Some(caps) = Regex::new(r"(\w+)\.\s*(\w*)$")
            .unwrap()
            .captures(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::ColumnName,
                partial_word: caps.get(2).map_or("", |m| m.as_str()).to_string(),
                table_qualifier: Some(caps.get(1).unwrap().as_str().to_string()),
            };
        }

        // CaseExpression context
        if Regex::new(r"(?i)\bCASE\s+(\w+\s+)?WHEN\s+\w*$")
            .unwrap()
            .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::CaseExpression,
                partial_word,
                table_qualifier: None,
            };
        }
        if Regex::new(r"(?i)\bCASE\b[^;]*\bTHEN\s+\w*$")
            .unwrap()
            .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::CaseExpression,
                partial_word,
                table_qualifier: None,
            };
        }

        // WindowFunction context - after OVER keyword
        if Regex::new(r"(?i)\bOVER\s*\(\s*$")
            .unwrap()
            .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::WindowFunction,
                partial_word: String::new(),
                table_qualifier: None,
            };
        }
        if Regex::new(r"(?i)\bOVER\s*\(\s*PARTITION\s+BY\s+\w*$")
            .unwrap()
            .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::ColumnName,
                partial_word,
                table_qualifier: None,
            };
        }
        if Regex::new(r"(?i)\bOVER\s*\(\s*ORDER\s+BY\s+\w*$")
            .unwrap()
            .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::ColumnName,
                partial_word,
                table_qualifier: None,
            };
        }

        // DataType context - after AS in CAST/CONVERT
        if Regex::new(r"(?i)\b(CAST|CONVERT|TRY_CAST)\s*\([^)]*\s+AS\s+(\w*)$")
            .unwrap()
            .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::DataType,
                partial_word,
                table_qualifier: None,
            };
        }

        // JoinCondition context - after JOIN...ON
        if Regex::new(r"(?i)\bJOIN\s+\w+\s+(AS\s+\w+\s+)?ON\s+\w*$")
            .unwrap()
            .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::JoinCondition,
                partial_word,
                table_qualifier: None,
            };
        }
        if Regex::new(r"(?i)\bJOIN\s+\w+\s+(AS\s+\w+\s+)?ON\s+[^=]+=\s*\w*$")
            .unwrap()
            .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::JoinCondition,
                partial_word,
                table_qualifier: None,
            };
        }

        // FunctionName context - after opening parenthesis with function-like pattern
        if Regex::new(r"(?i)\b(\w+)\s*\($")
            .unwrap()
            .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::FunctionName,
                partial_word: String::new(),
                table_qualifier: None,
            };
        }

        // TableName context - after FROM, JOIN, INTO, UPDATE
        if let Some(caps) = Regex::new(r"(?i)\b(FROM|JOIN|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|CROSS\s+JOIN|INTO|UPDATE)\s+(\w*)$").unwrap().captures(&text_before_cursor) {
            return QueryPosition {
                context: CompletionContext::TableName,
                partial_word: caps.get(2).map_or("", |m| m.as_str()).to_string(),
                table_qualifier: None,
            };
        }

        // ColumnName context - after SELECT
        if Regex::new(r"(?i)\bSELECT\s+(DISTINCT\s+)?(\w*)$")
            .unwrap()
            .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::ColumnName,
                partial_word,
                table_qualifier: None,
            };
        }

        // ColumnName context - after comma in SELECT (before FROM)
        if upper_text.contains("SELECT")
            && !upper_text.contains("FROM")
            && Regex::new(r",\s*(\w*)$")
                .unwrap()
                .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::ColumnName,
                partial_word,
                table_qualifier: None,
            };
        }

        // ColumnName context - in WHERE, HAVING, GROUP BY, ORDER BY clauses
        if Regex::new(r"(?i)\b(WHERE|HAVING|AND|OR)\s+(\w*)$")
            .unwrap()
            .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::ColumnName,
                partial_word,
                table_qualifier: None,
            };
        }
        if Regex::new(r"(?i)\b(GROUP\s+BY|ORDER\s+BY)\s+(\w*)$")
            .unwrap()
            .is_match(&text_before_cursor)
        {
            return QueryPosition {
                context: CompletionContext::ColumnName,
                partial_word,
                table_qualifier: None,
            };
        }

        // Default to Keyword context
        QueryPosition {
            context: CompletionContext::Keyword,
            partial_word,
            table_qualifier: None,
        }
    }

    fn get_sql_keywords(&self) -> Vec<&'static str> {
        vec![
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "MERGE",
            "COPY",
            "TRUNCATE",
            "CREATE",
            "ALTER",
            "DROP",
            "RENAME",
            "CLONE",
            "UNDROP",
            "TABLE",
            "VIEW",
            "MATERIALIZED VIEW",
            "SEQUENCE",
            "STAGE",
            "PIPE",
            "STREAM",
            "TASK",
            "DATABASE",
            "SCHEMA",
            "WAREHOUSE",
            "USER",
            "ROLE",
            "FROM",
            "WHERE",
            "GROUP BY",
            "HAVING",
            "ORDER BY",
            "LIMIT",
            "OFFSET",
            "QUALIFY",
            "SAMPLE",
            "TABLESAMPLE",
            "JOIN",
            "INNER JOIN",
            "LEFT JOIN",
            "RIGHT JOIN",
            "FULL JOIN",
            "FULL OUTER JOIN",
            "CROSS JOIN",
            "NATURAL JOIN",
            "LEFT OUTER JOIN",
            "RIGHT OUTER JOIN",
            "ON",
            "USING",
            "UNION",
            "UNION ALL",
            "INTERSECT",
            "EXCEPT",
            "MINUS",
            "CASE",
            "WHEN",
            "THEN",
            "ELSE",
            "END",
            "IF",
            "IFNULL",
            "NULLIF",
            "NVL",
            "IFF",
            "AND",
            "OR",
            "NOT",
            "IN",
            "NOT IN",
            "EXISTS",
            "NOT EXISTS",
            "BETWEEN",
            "LIKE",
            "ILIKE",
            "RLIKE",
            "REGEXP",
            "IS NULL",
            "IS NOT NULL",
            "IS TRUE",
            "IS FALSE",
            "DISTINCT",
            "ALL",
            "AS",
            "ASC",
            "DESC",
            "NULLS FIRST",
            "NULLS LAST",
            "WITH",
            "RECURSIVE",
            "LATERAL",
            "OVER",
            "PARTITION BY",
            "ROWS",
            "RANGE",
            "UNBOUNDED",
            "PRECEDING",
            "FOLLOWING",
            "CURRENT ROW",
            "VALUES",
            "DEFAULT",
            "NULL",
            "TRUE",
            "FALSE",
            "CAST",
            "CONVERT",
            "TRY_CAST",
            "TRY_TO_NUMBER",
            "TRY_TO_DATE",
            "TRY_TO_TIMESTAMP",
            "SHOW",
            "DESCRIBE",
            "DESC",
            "USE",
            "SET",
            "UNSET",
            "BEGIN",
            "COMMIT",
            "ROLLBACK",
            "TRANSACTION",
            "GRANT",
            "REVOKE",
            "EXECUTE",
            "USAGE",
            "OWNERSHIP",
        ]
    }

    fn get_aggregate_functions(&self) -> Vec<&'static str> {
        vec![
            "COUNT",
            "SUM",
            "AVG",
            "MIN",
            "MAX",
            "MEDIAN",
            "MODE",
            "STDDEV",
            "STDDEV_POP",
            "STDDEV_SAMP",
            "VARIANCE",
            "VAR_POP",
            "VAR_SAMP",
            "CORR",
            "COVAR_POP",
            "COVAR_SAMP",
            "LISTAGG",
            "ARRAY_AGG",
            "OBJECT_AGG",
            "APPROX_COUNT_DISTINCT",
            "APPROX_PERCENTILE",
            "APPROX_TOP_K",
            "ANY_VALUE",
            "BITAND_AGG",
            "BITOR_AGG",
            "BITXOR_AGG",
            "BOOLAND_AGG",
            "BOOLOR_AGG",
            "BOOLXOR_AGG",
        ]
    }

    fn get_window_functions(&self) -> Vec<&'static str> {
        vec![
            "ROW_NUMBER",
            "RANK",
            "DENSE_RANK",
            "PERCENT_RANK",
            "CUME_DIST",
            "NTILE",
            "LAG",
            "LEAD",
            "FIRST_VALUE",
            "LAST_VALUE",
            "NTH_VALUE",
            "RATIO_TO_REPORT",
            "WIDTH_BUCKET",
        ]
    }

    fn get_string_functions(&self) -> Vec<&'static str> {
        vec![
            "CONCAT",
            "CONCAT_WS",
            "LENGTH",
            "LEN",
            "CHAR_LENGTH",
            "CHARACTER_LENGTH",
            "LOWER",
            "UPPER",
            "INITCAP",
            "TRIM",
            "LTRIM",
            "RTRIM",
            "SUBSTR",
            "SUBSTRING",
            "LEFT",
            "RIGHT",
            "SPLIT",
            "SPLIT_PART",
            "REPLACE",
            "REGEXP_REPLACE",
            "REGEXP_SUBSTR",
            "REGEXP_INSTR",
            "REGEXP_COUNT",
            "CONTAINS",
            "STARTSWITH",
            "ENDSWITH",
            "POSITION",
            "CHARINDEX",
            "REVERSE",
            "REPEAT",
            "SPACE",
            "LPAD",
            "RPAD",
            "TRANSLATE",
            "SOUNDEX",
            "EDITDISTANCE",
            "JAROWINKLER_SIMILARITY",
            "ASCII",
            "CHR",
            "UNICODE",
            "HEX_ENCODE",
            "HEX_DECODE",
            "BASE64_ENCODE",
            "BASE64_DECODE",
            "MD5",
            "SHA1",
            "SHA2",
            "HASH",
        ]
    }

    fn get_date_time_functions(&self) -> Vec<&'static str> {
        vec![
            "CURRENT_DATE",
            "CURRENT_TIME",
            "CURRENT_TIMESTAMP",
            "SYSDATE",
            "SYSTIMESTAMP",
            "GETDATE",
            "LOCALTIME",
            "LOCALTIMESTAMP",
            "DATE_TRUNC",
            "TRUNC",
            "DATE_PART",
            "DATEADD",
            "DATEDIFF",
            "TIMESTAMPADD",
            "TIMESTAMPDIFF",
            "YEAR",
            "QUARTER",
            "MONTH",
            "WEEK",
            "WEEKOFYEAR",
            "DAYOFWEEK",
            "DAYOFWEEKISO",
            "DAYOFYEAR",
            "DAY",
            "DAYNAME",
            "MONTHNAME",
            "HOUR",
            "MINUTE",
            "SECOND",
            "LAST_DAY",
            "NEXT_DAY",
            "PREVIOUS_DAY",
            "ADD_MONTHS",
            "TO_DATE",
            "TO_TIME",
            "TO_TIMESTAMP",
            "TO_TIMESTAMP_LTZ",
            "TO_TIMESTAMP_NTZ",
            "TO_TIMESTAMP_TZ",
            "TRY_TO_DATE",
            "TRY_TO_TIME",
            "TRY_TO_TIMESTAMP",
            "DATE_FROM_PARTS",
            "TIME_FROM_PARTS",
            "TIMESTAMP_FROM_PARTS",
            "EXTRACT",
            "TIME_SLICE",
            "CONVERT_TIMEZONE",
        ]
    }

    fn get_numeric_functions(&self) -> Vec<&'static str> {
        vec![
            "ABS",
            "CEIL",
            "CEILING",
            "FLOOR",
            "ROUND",
            "TRUNCATE",
            "TRUNC",
            "MOD",
            "POWER",
            "POW",
            "SQRT",
            "EXP",
            "LN",
            "LOG",
            "SIGN",
            "DEGREES",
            "RADIANS",
            "SIN",
            "COS",
            "TAN",
            "ASIN",
            "ACOS",
            "ATAN",
            "ATAN2",
            "SINH",
            "COSH",
            "TANH",
            "ASINH",
            "ACOSH",
            "ATANH",
            "PI",
            "RANDOM",
            "UNIFORM",
            "NORMAL",
            "ZIPF",
            "DIV0",
            "DIV0NULL",
            "GREATEST",
            "LEAST",
            "BITAND",
            "BITOR",
            "BITXOR",
            "BITNOT",
            "BITSHIFTLEFT",
            "BITSHIFTRIGHT",
            "GETBIT",
            "HAVERSINE",
            "SQUARE",
        ]
    }

    fn get_conversion_functions(&self) -> Vec<&'static str> {
        vec![
            "TO_CHAR",
            "TO_VARCHAR",
            "TO_NUMBER",
            "TO_DECIMAL",
            "TO_NUMERIC",
            "TO_DOUBLE",
            "TO_BINARY",
            "TO_BOOLEAN",
            "TO_ARRAY",
            "TO_OBJECT",
            "TO_VARIANT",
            "TRY_TO_BINARY",
            "TRY_TO_BOOLEAN",
            "TRY_TO_NUMBER",
            "TRY_TO_DECIMAL",
            "TRY_TO_NUMERIC",
            "TRY_TO_DOUBLE",
            "PARSE_JSON",
            "PARSE_XML",
            "TRY_PARSE_JSON",
        ]
    }

    fn get_conditional_functions(&self) -> Vec<&'static str> {
        vec![
            "COALESCE",
            "DECODE",
            "IFF",
            "IFNULL",
            "NVL",
            "NVL2",
            "NULLIF",
            "ZEROIFNULL",
            "EQUAL_NULL",
            "REGR_VALX",
            "REGR_VALY",
        ]
    }

    fn get_semi_structured_functions(&self) -> Vec<&'static str> {
        vec![
            "ARRAY_CONSTRUCT",
            "ARRAY_SIZE",
            "ARRAY_SLICE",
            "ARRAY_APPEND",
            "ARRAY_CAT",
            "ARRAY_COMPACT",
            "ARRAY_CONTAINS",
            "ARRAY_INSERT",
            "ARRAY_INTERSECTION",
            "ARRAY_POSITION",
            "ARRAY_PREPEND",
            "ARRAY_REMOVE",
            "ARRAY_TO_STRING",
            "ARRAY_UNIQUE_AGG",
            "ARRAYS_OVERLAP",
            "OBJECT_CONSTRUCT",
            "OBJECT_DELETE",
            "OBJECT_INSERT",
            "OBJECT_PICK",
            "OBJECT_KEYS",
            "GET",
            "GET_PATH",
            "FLATTEN",
            "PARSE_JSON",
            "CHECK_JSON",
            "CHECK_XML",
            "STRIP_NULL_VALUE",
            "XMLGET",
            "AS_ARRAY",
            "AS_BINARY",
            "AS_BOOLEAN",
            "AS_CHAR",
            "AS_DATE",
            "AS_DECIMAL",
            "AS_DOUBLE",
            "AS_INTEGER",
            "AS_NUMBER",
            "AS_OBJECT",
            "AS_REAL",
            "AS_TIME",
            "AS_TIMESTAMP_LTZ",
            "AS_TIMESTAMP_NTZ",
            "AS_TIMESTAMP_TZ",
            "AS_VARCHAR",
        ]
    }

    fn get_context_functions(&self) -> Vec<&'static str> {
        vec![
            "CURRENT_ACCOUNT",
            "CURRENT_CLIENT",
            "CURRENT_DATABASE",
            "CURRENT_ROLE",
            "CURRENT_SCHEMA",
            "CURRENT_SCHEMAS",
            "CURRENT_SESSION",
            "CURRENT_STATEMENT",
            "CURRENT_TRANSACTION",
            "CURRENT_USER",
            "CURRENT_VERSION",
            "CURRENT_WAREHOUSE",
            "INVOKER_ROLE",
            "INVOKER_SHARE",
            "IS_GRANTED_TO_INVOKER_ROLE",
            "IS_ROLE_IN_SESSION",
        ]
    }

    fn get_data_types(&self) -> Vec<&'static str> {
        vec![
            "NUMBER",
            "DECIMAL",
            "NUMERIC",
            "INT",
            "INTEGER",
            "BIGINT",
            "SMALLINT",
            "TINYINT",
            "BYTEINT",
            "FLOAT",
            "FLOAT4",
            "FLOAT8",
            "DOUBLE",
            "DOUBLE PRECISION",
            "REAL",
            "VARCHAR",
            "CHAR",
            "CHARACTER",
            "STRING",
            "TEXT",
            "BINARY",
            "VARBINARY",
            "BOOLEAN",
            "DATE",
            "DATETIME",
            "TIME",
            "TIMESTAMP",
            "TIMESTAMP_LTZ",
            "TIMESTAMP_NTZ",
            "TIMESTAMP_TZ",
            "VARIANT",
            "OBJECT",
            "ARRAY",
            "GEOGRAPHY",
            "GEOMETRY",
        ]
    }

    fn calculate_completion_score(&self, candidate: &str, partial: &str) -> i32 {
        if partial.is_empty() {
            return 100;
        }
        let candidate_upper = candidate.to_uppercase();
        let partial_upper = partial.to_uppercase();

        if candidate_upper == partial_upper {
            return 1000;
        }
        if candidate_upper.starts_with(&partial_upper) {
            return 500 - partial.len() as i32;
        }
        if candidate_upper.contains(&partial_upper) {
            return 100;
        }
        0
    }

    fn extract_tables_from_query(&self, text: &str) -> Vec<String> {
        let mut tables = Vec::new();
        let re = Regex::new(r"(?i)\b(?:FROM|JOIN|INTO|UPDATE)\s+(\w+)").unwrap();
        for caps in re.captures_iter(text) {
            if let Some(table) = caps.get(1) {
                tables.push(table.as_str().to_string());
            }
        }
        tables
    }

    fn format_sql(&self, sql: &str) -> String {
        let mut formatted = sql.to_string();

        // Add newlines before major keywords
        for keyword in &[
            "FROM",
            "WHERE",
            "JOIN",
            "INNER JOIN",
            "LEFT JOIN",
            "RIGHT JOIN",
            "FULL JOIN",
            "GROUP BY",
            "ORDER BY",
            "HAVING",
            "LIMIT",
            "OFFSET",
        ] {
            let pattern = format!(r"(?i)\s+({})\s+", regex::escape(keyword));
            if let Ok(re) = Regex::new(&pattern) {
                formatted = re
                    .replace_all(&formatted, |caps: &regex::Captures| {
                        format!("\n{} ", caps.get(1).unwrap().as_str().to_uppercase())
                    })
                    .to_string();
            }
        }

        // Uppercase all SQL keywords
        for keyword in self.get_sql_keywords() {
            let pattern = format!(r"(?i)\b{}\b", regex::escape(keyword));
            if let Ok(re) = Regex::new(&pattern) {
                formatted = re.replace_all(&formatted, keyword).to_string();
            }
        }

        // Basic indentation
        let lines: Vec<&str> = formatted.lines().collect();
        let mut indented = Vec::new();
        let mut indent_level = 0;

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Decrease indent for certain keywords
            if trimmed.starts_with("FROM")
                || trimmed.starts_with("WHERE")
                || trimmed.starts_with("GROUP BY")
                || trimmed.starts_with("ORDER BY")
                || trimmed.starts_with("HAVING")
            {
                indent_level = 1;
            } else if trimmed.starts_with("SELECT") {
                indent_level = 0;
            }

            indented.push(format!("{}{}", "  ".repeat(indent_level), trimmed));
        }

        indented.join("\n")
    }

    async fn get_cortex_suggestions(&self, text: &str, position: Position) -> Vec<CompletionItem> {
        let mut engine_lock = self.engine.lock().await; // Changed to mut
        if let Some(engine) = &mut *engine_lock {
            let lines: Vec<&str> = text.lines().collect();
            let current_query = if (position.line as usize) < lines.len() {
                lines[..=(position.line as usize)].join("\n")
            } else {
                text.to_string()
            };

            match engine.get_query_suggestions(&current_query).await {
                Ok(suggestions) => suggestions
                    .into_iter()
                    .map(|suggestion| CompletionItem {
                        label: suggestion.clone(),
                        kind: Some(CompletionItemKind::TEXT),
                        detail: Some("Cortex AI Suggestion".to_string()),
                        insert_text: Some(suggestion),
                        insert_text_format: Some(InsertTextFormat::PLAIN_TEXT),
                        ..Default::default()
                    })
                    .collect(),
                Err(_) => vec![],
            }
        } else {
            vec![]
        }
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            server_info: Some(ServerInfo {
                name: "snowflake-lsp".to_string(),
                version: Some("1.0.0".to_string()),
            }),
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                completion_provider: Some(CompletionOptions {
                    resolve_provider: Some(false),
                    trigger_characters: Some(vec![
                        ".".to_string(),
                        " ".to_string(),
                        ",".to_string(),
                        "(".to_string(),
                    ]),
                    work_done_progress_options: Default::default(),
                    all_commit_characters: None,
                    completion_item: None,
                }),
                execute_command_provider: Some(ExecuteCommandOptions {
                    commands: vec![
                        "snowflake.generateQuery".to_string(),
                        "snowflake.uploadSemanticModel".to_string(),
                        "snowflake.verifySemanticModel".to_string(),
                        "snowflake.generateSemanticModel".to_string(),
                        "snowflake.chatWithAnalyst".to_string(),
                    ],
                    work_done_progress_options: Default::default(),
                }),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                document_symbol_provider: Some(OneOf::Left(true)),
                document_formatting_provider: Some(OneOf::Left(true)),
                signature_help_provider: Some(SignatureHelpOptions {
                    trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
                    retrigger_characters: None,
                    work_done_progress_options: Default::default(),
                }),
                definition_provider: Some(OneOf::Left(true)),
                workspace: Some(WorkspaceServerCapabilities {
                    workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                        supported: Some(true),
                        change_notifications: Some(OneOf::Left(true)),
                    }),
                    file_operations: None,
                }),
                ..ServerCapabilities::default()
            },
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "Snowflake LSP initialized!")
            .await;

        let engine_lock = self.engine.lock().await;
        if let Some(engine) = &*engine_lock {
            match engine.get_detailed_schema().await {
                Ok(schema) => {
                    *self.schema_cache.lock().await = schema;
                    let count = self.schema_cache.lock().await.len();
                    self.client
                        .log_message(
                            MessageType::INFO,
                            &format!("Schema cached: {} columns", count),
                        )
                        .await;
                }
                Err(e) => {
                    self.client
                        .log_message(MessageType::ERROR, &format!("Schema fetch failed: {}", e))
                        .await;
                }
            }
        }
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.document_map.insert(
            params.text_document.uri.to_string(),
            params.text_document.text,
        );
    }

    async fn did_change(&self, mut params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri.to_string();
        let text = params.content_changes.remove(0).text;
        self.document_map.insert(uri.clone(), text.clone());

        let engine_lock = self.engine.lock().await;
        if let Some(engine) = &*engine_lock {
            let diagnostics = match engine.validate_query(&text).await {
                Ok(_) => vec![],
                Err(e) => {
                    let re = Regex::new(r"line (\d+)").unwrap();
                    let mut range = Range::new(Position::new(0, 0), Position::new(0, 1000));
                    if let Some(caps) = re.captures(&e) {
                        if let Some(line_match) = caps.get(1) {
                            if let Ok(line) = line_match.as_str().parse::<u32>() {
                                let line = if line > 0 { line - 1 } else { 0 };
                                range =
                                    Range::new(Position::new(line, 0), Position::new(line, 1000));
                            }
                        }
                    }
                    vec![Diagnostic::new_simple(range, e)]
                }
            };
            self.client
                .publish_diagnostics(params.text_document.uri, diagnostics, None)
                .await;
        }
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let uri = params.text_document_position.text_document.uri.to_string();
        let position = params.text_document_position.position;

        let document = match self.document_map.get(&uri) {
            Some(doc) => doc,
            None => return Ok(None),
        };
        let text = document.value();

        let query_pos = self.analyze_query_context(text, position);
        let partial_word = &query_pos.partial_word;

        self.client
            .log_message(
                MessageType::INFO,
                &format!(
                    "Context: {:?}, partial: '{}'",
                    query_pos.context, partial_word
                ),
            )
            .await;

        let mut items = Vec::new();
        let schema_cache = self.schema_cache.lock().await;

        // Add Cortex AI suggestions at the top
        let cortex_suggestions = self.get_cortex_suggestions(text, position).await;
        for (idx, suggestion) in cortex_suggestions.into_iter().enumerate() {
            items.push((2000 + idx as i32, suggestion));
        }

        match query_pos.context {
            CompletionContext::TableName => {
                let mut seen = std::collections::HashSet::new();
                for item in schema_cache.iter() {
                    if seen.insert(item.table_name.clone()) {
                        let score = self.calculate_completion_score(&item.table_name, partial_word);
                        if score > 0 {
                            items.push((
                                score,
                                CompletionItem {
                                    label: item.table_name.clone(),
                                    kind: Some(CompletionItemKind::CLASS),
                                    detail: Some(format!(
                                        "{}.{}",
                                        item.table_schema, item.table_name
                                    )),
                                    ..Default::default()
                                },
                            ));
                        }
                    }
                }
            }
            CompletionContext::ColumnName => {
                if let Some(table) = &query_pos.table_qualifier {
                    for item in schema_cache.iter() {
                        if item.table_name.eq_ignore_ascii_case(table) {
                            let score =
                                self.calculate_completion_score(&item.column_name, partial_word);
                            if score > 0 {
                                items.push((
                                    score,
                                    CompletionItem {
                                        label: item.column_name.clone(),
                                        kind: Some(CompletionItemKind::FIELD),
                                        detail: Some(format!(
                                            "{} ({})",
                                            item.data_type, item.table_name
                                        )),
                                        ..Default::default()
                                    },
                                ));
                            }
                        }
                    }
                } else {
                    let tables_in_query = self.extract_tables_from_query(text);
                    let mut seen = std::collections::HashSet::new();
                    for item in schema_cache.iter() {
                        let key = format!("{}:{}", item.column_name, item.data_type);
                        if seen.insert(key) {
                            let score =
                                self.calculate_completion_score(&item.column_name, partial_word);
                            if score > 0 {
                                let in_query = tables_in_query
                                    .iter()
                                    .any(|t| t.eq_ignore_ascii_case(&item.table_name));
                                let final_score = if in_query { score + 300 } else { score };
                                items.push((
                                    final_score,
                                    CompletionItem {
                                        label: item.column_name.clone(),
                                        kind: Some(CompletionItemKind::FIELD),
                                        detail: Some(format!(
                                            "{} ({})",
                                            item.data_type, item.table_name
                                        )),
                                        ..Default::default()
                                    },
                                ));
                            }
                        }
                    }
                }
            }
            CompletionContext::FunctionName | CompletionContext::CaseExpression => {
                for func in self
                    .get_aggregate_functions()
                    .iter()
                    .chain(self.get_string_functions().iter())
                    .chain(self.get_date_time_functions().iter())
                    .chain(self.get_numeric_functions().iter())
                    .chain(self.get_conditional_functions().iter())
                    .chain(self.get_conversion_functions().iter())
                    .chain(self.get_semi_structured_functions().iter())
                    .chain(self.get_context_functions().iter())
                {
                    let score = self.calculate_completion_score(func, partial_word);
                    if score > 0 {
                        items.push((
                            score,
                            CompletionItem {
                                label: func.to_string(),
                                kind: Some(CompletionItemKind::FUNCTION),
                                detail: Some("Function".to_string()),
                                insert_text: Some(format!("{}(${{1}})", func)),
                                insert_text_format: Some(InsertTextFormat::SNIPPET),
                                ..Default::default()
                            },
                        ));
                    }
                }
            }
            CompletionContext::DataType => {
                for dtype in self.get_data_types() {
                    let score = self.calculate_completion_score(dtype, partial_word);
                    if score > 0 {
                        items.push((
                            score,
                            CompletionItem {
                                label: dtype.to_string(),
                                kind: Some(CompletionItemKind::TYPE_PARAMETER),
                                detail: Some("Data Type".to_string()),
                                ..Default::default()
                            },
                        ));
                    }
                }
            }
            CompletionContext::WindowFunction => {
                for func in self.get_window_functions() {
                    let score = self.calculate_completion_score(func, partial_word);
                    if score > 0 {
                        items.push((
                            score,
                            CompletionItem {
                                label: format!("{} PARTITION BY", func),
                                kind: Some(CompletionItemKind::FUNCTION),
                                detail: Some("Window Function".to_string()),
                                insert_text: Some("PARTITION BY ${1:column}".to_string()),
                                insert_text_format: Some(InsertTextFormat::SNIPPET),
                                ..Default::default()
                            },
                        ));
                    }
                }
                items.push((
                    1000,
                    CompletionItem {
                        label: "ORDER BY".to_string(),
                        kind: Some(CompletionItemKind::KEYWORD),
                        insert_text: Some("ORDER BY ${1:column}".to_string()),
                        insert_text_format: Some(InsertTextFormat::SNIPPET),
                        ..Default::default()
                    },
                ));
            }
            CompletionContext::JoinCondition => {
                let tables_in_query = self.extract_tables_from_query(text);
                let mut seen = std::collections::HashSet::new();
                for item in schema_cache.iter() {
                    if tables_in_query
                        .iter()
                        .any(|t| t.eq_ignore_ascii_case(&item.table_name))
                    {
                        let key = item.column_name.clone();
                        if seen.insert(key) {
                            let score =
                                self.calculate_completion_score(&item.column_name, partial_word);
                            if score > 0 {
                                items.push((
                                    score + 500,
                                    CompletionItem {
                                        label: format!("{}.{}", item.table_name, item.column_name),
                                        kind: Some(CompletionItemKind::FIELD),
                                        detail: Some(item.data_type.to_string()),
                                        ..Default::default()
                                    },
                                ));
                            }
                        }
                    }
                }
            }
            CompletionContext::SubQuery => {
                for kw in &["SELECT", "WITH", "FROM", "WHERE", "GROUP BY", "ORDER BY"] {
                    items.push((
                        500,
                        CompletionItem {
                            label: kw.to_string(),
                            kind: Some(CompletionItemKind::KEYWORD),
                            ..Default::default()
                        },
                    ));
                }
            }
            CompletionContext::Keyword => {
                for kw in self.get_sql_keywords() {
                    let score = self.calculate_completion_score(kw, partial_word);
                    if score > 0 {
                        items.push((
                            score,
                            CompletionItem {
                                label: kw.to_string(),
                                kind: Some(CompletionItemKind::KEYWORD),
                                ..Default::default()
                            },
                        ));
                    }
                }

                let mut seen = std::collections::HashSet::new();
                for item in schema_cache.iter() {
                    if seen.insert(item.table_name.clone()) {
                        let score = self.calculate_completion_score(&item.table_name, partial_word);
                        if score > 0 {
                            items.push((
                                score,
                                CompletionItem {
                                    label: item.table_name.clone(),
                                    kind: Some(CompletionItemKind::CLASS),
                                    detail: Some("Table".to_string()),
                                    ..Default::default()
                                },
                            ));
                        }
                    }
                }
            }
        }

        items.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.label.cmp(&b.1.label)));
        let completion_items: Vec<CompletionItem> =
            items.into_iter().map(|(_, item)| item).take(200).collect();

        self.client
            .log_message(
                MessageType::INFO,
                &format!("Returning {} completions", completion_items.len()),
            )
            .await;
        Ok(Some(CompletionResponse::Array(completion_items)))
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        let uri = params
            .text_document_position_params
            .text_document
            .uri
            .to_string();
        let position = params.text_document_position_params.position;

        let document = match self.document_map.get(&uri) {
            Some(doc) => doc,
            None => return Ok(None),
        };
        let text = document.value();

        let (word, range) = self.get_word_at_position(text, position);
        if word.is_empty() {
            return Ok(None);
        }

        let schema_cache = self.schema_cache.lock().await;

        // Check if it's a table
        let mut table_columns = Vec::new();
        for item in schema_cache.iter() {
            if item.table_name.eq_ignore_ascii_case(&word) {
                table_columns.push(format!("  - {} ({})", item.column_name, item.data_type));
            }
        }

        if !table_columns.is_empty() {
            let content = format!(
                "**Table: {}**\n\nColumns:\n{}",
                word,
                table_columns.join("\n")
            );
            return Ok(Some(Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: content,
                }),
                range: Some(range),
            }));
        }

        // Check if it's a column
        for item in schema_cache.iter() {
            if item.column_name.eq_ignore_ascii_case(&word) {
                let content = format!(
                    "**Column: {}**\n\nTable: {}\nType: {}",
                    item.column_name, item.table_name, item.data_type
                );
                return Ok(Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: content,
                    }),
                    range: Some(range),
                }));
            }
        }

        Ok(None)
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        let uri = params
            .text_document_position_params
            .text_document
            .uri
            .to_string();
        let position = params.text_document_position_params.position;

        let document = match self.document_map.get(&uri) {
            Some(doc) => doc,
            None => return Ok(None),
        };
        let text = document.value();

        let (word, _) = self.get_word_at_position(text, position);
        if word.is_empty() {
            return Ok(None);
        }

        let schema_cache = self.schema_cache.lock().await;

        // Find first occurrence of the table or column
        for item in schema_cache.iter() {
            if item.table_name.eq_ignore_ascii_case(&word)
                || item.column_name.eq_ignore_ascii_case(&word)
            {
                // Return the location in the document where it's first used
                let re = Regex::new(&format!(r"(?i)\b{}\b", regex::escape(&word))).unwrap();
                if let Some(mat) = re.find(text) {
                    let start = mat.start();
                    let lines_before = text[..start].matches('\n').count();
                    let line_start = text[..start].rfind('\n').map(|i| i + 1).unwrap_or(0);
                    let char_offset = start - line_start;

                    let location = Location {
                        uri: params
                            .text_document_position_params
                            .text_document
                            .uri
                            .clone(),
                        range: Range::new(
                            Position::new(lines_before as u32, char_offset as u32),
                            Position::new(lines_before as u32, (char_offset + word.len()) as u32),
                        ),
                    };
                    return Ok(Some(GotoDefinitionResponse::Scalar(location)));
                }
            }
        }

        Ok(None)
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        let uri = params.text_document.uri.to_string();

        let document = match self.document_map.get(&uri) {
            Some(doc) => doc,
            None => return Ok(None),
        };
        let text = document.value();

        let mut symbols = Vec::new();
        let tables = self.extract_tables_from_query(text);
        let schema_cache = self.schema_cache.lock().await;

        for (idx, table) in tables.iter().enumerate() {
            let range = Range::new(Position::new(idx as u32, 0), Position::new(idx as u32, 100));

            let mut children = Vec::new();
            for item in schema_cache.iter() {
                if item.table_name.eq_ignore_ascii_case(table) {
                    #[allow(deprecated)]
                    children.push(DocumentSymbol {
                        name: item.column_name.clone(),
                        detail: Some(item.data_type.clone()),
                        kind: SymbolKind::FIELD,
                        range,
                        selection_range: range,
                        children: None,
                        tags: None,
                        deprecated: None,
                    });
                }
            }

            #[allow(deprecated)]
            symbols.push(DocumentSymbol {
                name: table.clone(),
                detail: Some("Table".to_string()),
                kind: SymbolKind::CLASS,
                range,
                selection_range: range,
                children: Some(children),
                tags: None,
                deprecated: None,
            });
        }

        Ok(Some(DocumentSymbolResponse::Nested(symbols)))
    }

    async fn formatting(&self, params: DocumentFormattingParams) -> Result<Option<Vec<TextEdit>>> {
        let uri = params.text_document.uri.to_string();

        let document = match self.document_map.get(&uri) {
            Some(doc) => doc,
            None => return Ok(None),
        };
        let text = document.value();

        let formatted = self.format_sql(text);

        let line_count = text.lines().count();
        let last_line_length = text.lines().last().map(|l| l.len()).unwrap_or(0);

        Ok(Some(vec![TextEdit {
            range: Range::new(
                Position::new(0, 0),
                Position::new(line_count as u32, last_line_length as u32),
            ),
            new_text: formatted,
        }]))
    }

    async fn signature_help(&self, params: SignatureHelpParams) -> Result<Option<SignatureHelp>> {
        let uri = params
            .text_document_position_params
            .text_document
            .uri
            .to_string();
        let position = params.text_document_position_params.position;

        let document = match self.document_map.get(&uri) {
            Some(doc) => doc,
            None => return Ok(None),
        };
        let text = document.value();

        // Extract text up to cursor
        let lines: Vec<&str> = text.lines().collect();
        let mut text_before = String::new();
        for (i, line) in lines.iter().enumerate() {
            if i < position.line as usize {
                text_before.push_str(line);
                text_before.push(' ');
            } else if i == position.line as usize {
                let end = (position.character as usize).min(line.len());
                text_before.push_str(&line[..end]);
                break;
            }
        }

        // Check for common SQL functions
        let function_re = Regex::new(r"(?i)(\w+)\s*\([^)]*$").unwrap();
        if let Some(caps) = function_re.captures(&text_before) {
            let func_name = caps.get(1).unwrap().as_str().to_uppercase();

            let (label, doc) = match func_name.as_str() {
                "COUNT" => ("COUNT(column)", "Returns the number of rows"),
                "SUM" => ("SUM(column)", "Returns the sum of values"),
                "AVG" => ("AVG(column)", "Returns the average of values"),
                "MIN" => ("MIN(column)", "Returns the minimum value"),
                "MAX" => ("MAX(column)", "Returns the maximum value"),
                "CAST" => ("CAST(value AS type)", "Converts value to specified type"),
                "COALESCE" => ("COALESCE(val1, val2, ...)", "Returns first non-null value"),
                "DATEADD" => ("DATEADD(unit, amount, date)", "Adds time to a date"),
                "DATEDIFF" => (
                    "DATEDIFF(unit, start_date, end_date)",
                    "Calculates difference between dates",
                ),
                _ => return Ok(None),
            };

            return Ok(Some(SignatureHelp {
                signatures: vec![SignatureInformation {
                    label: label.to_string(),
                    documentation: Some(Documentation::String(doc.to_string())),
                    parameters: None,
                    active_parameter: None,
                }],
                active_signature: Some(0),
                active_parameter: None,
            }));
        }

        Ok(None)
    }

    async fn execute_command(
        &self,
        params: ExecuteCommandParams,
    ) -> Result<Option<serde_json::Value>> {
        match params.command.as_str() {
            "snowflake.generateQuery" => {
                if let Some(natural_language) = params.arguments.first().and_then(|v| v.as_str()) {
                    let mut engine_lock = self.engine.lock().await; // Changed to mut
                    if let Some(engine) = &mut *engine_lock {
                        match engine
                            .execute_with_cortex_context(natural_language, false)
                            .await
                        {
                            Ok(result) => {
                                self.client
                                    .log_message(MessageType::INFO, "Query generated successfully")
                                    .await;
                                return Ok(Some(result));
                            }
                            Err(e) => {
                                self.client
                                    .log_message(
                                        MessageType::ERROR,
                                        &format!("Cortex error: {}", e),
                                    )
                                    .await;
                                return Ok(Some(serde_json::json!({ "error": e })));
                            }
                        }
                    }
                }
            }
            "snowflake.uploadSemanticModel" => {
                if let Some(path_str) = params.arguments.first().and_then(|v| v.as_str()) {
                    let mut engine_lock = self.engine.lock().await;
                    if let Some(engine) = &mut *engine_lock {
                        let path = std::path::PathBuf::from(path_str);
                        let stage_name = params
                            .arguments
                            .get(1)
                            .and_then(|v| v.as_str())
                            .unwrap_or("cortex_analyst_semantic_models");

                        match engine.upload_semantic_model(&path, stage_name).await {
                            Ok(msg) => {
                                self.client.log_message(MessageType::INFO, &msg).await;
                                return Ok(Some(
                                    serde_json::json!({ "success": true, "message": msg }),
                                ));
                            }
                            Err(e) => {
                                self.client
                                    .log_message(
                                        MessageType::ERROR,
                                        &format!("Upload failed: {}", e),
                                    )
                                    .await;
                                return Ok(Some(serde_json::json!({ "error": e })));
                            }
                        }
                    }
                }
            }
            "snowflake.verifySemanticModel" => {
                let engine_lock = self.engine.lock().await;
                if let Some(engine) = &*engine_lock {
                    match engine.verify_semantic_model().await {
                        Ok(warnings) => {
                            if warnings.is_empty() {
                                self.client
                                    .log_message(
                                        MessageType::INFO,
                                        "Semantic model verified successfully",
                                    )
                                    .await;
                            } else {
                                for warning in &warnings {
                                    self.client.log_message(MessageType::WARNING, warning).await;
                                }
                            }
                            return Ok(Some(serde_json::json!({ "warnings": warnings })));
                        }
                        Err(e) => {
                            self.client
                                .log_message(
                                    MessageType::ERROR,
                                    &format!("Verification failed: {}", e),
                                )
                                .await;
                            return Ok(Some(serde_json::json!({ "error": e })));
                        }
                    }
                }
            }
            "snowflake.generateSemanticModel" => {
                let engine_lock = self.engine.lock().await;
                if let Some(engine) = &*engine_lock {
                    match engine.generate_semantic_model_from_schema().await {
                        Ok(model) => {
                            self.client
                                .log_message(MessageType::INFO, "Semantic model generated")
                                .await;
                            return Ok(Some(serde_json::to_value(&model).unwrap_or_default()));
                        }
                        Err(e) => {
                            self.client
                                .log_message(
                                    MessageType::ERROR,
                                    &format!("Generation failed: {}", e),
                                )
                                .await;
                            return Ok(Some(serde_json::json!({ "error": e })));
                        }
                    }
                }
            }
            "snowflake.chatWithAnalyst" => {
                if let Some(message) = params.arguments.first().and_then(|v| v.as_str()) {
                    let use_history = params
                        .arguments
                        .get(1)
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);

                    let mut engine_lock = self.engine.lock().await; // Changed to mut
                    if let Some(engine) = &mut *engine_lock {
                        match engine.chat_with_analyst(message, use_history).await {
                            Ok(response) => {
                                return Ok(Some(
                                    serde_json::to_value(&response).unwrap_or_default(),
                                ));
                            }
                            Err(e) => {
                                self.client
                                    .log_message(MessageType::ERROR, &format!("Chat failed: {}", e))
                                    .await;
                                return Ok(Some(serde_json::json!({ "error": e })));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(None)
    }
}

pub async fn start_lsp(engine: Option<SnowflakeEngine>) {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| Backend {
        client,
        engine: Arc::new(Mutex::new(engine)),
        document_map: Arc::new(DashMap::new()),
        schema_cache: Arc::new(Mutex::new(Vec::new())),
    });
    Server::new(stdin, stdout, socket).serve(service).await;
}
