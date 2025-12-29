//! Document job tracking and cancellation for the CLI.
//!
//! Tracks document upload/parse jobs in Snowflake tables with status,
//! progress, and cancellation support.

use crate::engine::SnowflakeEngine;
use crate::events::EventEmitter;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

// ============================================================================
// Job Types
// ============================================================================

/// Job status values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Processing,
    EmbeddingText,
    EmbeddingImages,
    Complete,
    Failed,
    Cancelled,
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobStatus::Pending => write!(f, "pending"),
            JobStatus::Processing => write!(f, "processing"),
            JobStatus::EmbeddingText => write!(f, "embedding_text"),
            JobStatus::EmbeddingImages => write!(f, "embedding_images"),
            JobStatus::Complete => write!(f, "complete"),
            JobStatus::Failed => write!(f, "failed"),
            JobStatus::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl JobStatus {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "pending" => JobStatus::Pending,
            "processing" => JobStatus::Processing,
            "embedding_text" => JobStatus::EmbeddingText,
            "embedding_images" => JobStatus::EmbeddingImages,
            "complete" => JobStatus::Complete,
            "failed" => JobStatus::Failed,
            "cancelled" => JobStatus::Cancelled,
            _ => JobStatus::Pending,
        }
    }
}

/// Document job information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentJob {
    pub job_id: String,
    pub file_name: String,
    pub stage_path: String,
    pub status: JobStatus,
    pub progress: Option<String>,
    pub error_message: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
}

// ============================================================================
// Job Manager
// ============================================================================

/// In-memory tracking of active jobs with cancellation tokens
struct ActiveJob {
    job_id: String,
    file_name: String,
    cancel_token: CancellationToken,
}

/// Job manager for document processing tasks
pub struct JobManager {
    /// In-memory map of active jobs (ephemeral per session)
    active_jobs: HashMap<String, ActiveJob>,
    /// Default timeout in seconds
    default_timeout: u64,
}

impl JobManager {
    pub fn new() -> Self {
        Self {
            active_jobs: HashMap::new(),
            default_timeout: 120,
        }
    }

    /// Set the default timeout for operations
    pub fn set_timeout(&mut self, seconds: u64) {
        self.default_timeout = seconds;
    }

    /// Get the default timeout
    pub fn get_timeout(&self) -> u64 {
        self.default_timeout
    }

    /// Create a new job and register it
    pub fn create_job(&mut self, file_name: &str) -> (String, CancellationToken) {
        let job_id = Uuid::new_v4().to_string();
        let cancel_token = CancellationToken::new();

        self.active_jobs.insert(
            job_id.clone(),
            ActiveJob {
                job_id: job_id.clone(),
                file_name: file_name.to_string(),
                cancel_token: cancel_token.clone(),
            },
        );

        (job_id, cancel_token)
    }

    /// Get cancellation token for a job
    pub fn get_cancel_token(&self, job_id: &str) -> Option<CancellationToken> {
        self.active_jobs.get(job_id).map(|j| j.cancel_token.clone())
    }

    /// Cancel a job by ID
    pub fn cancel_job(&mut self, job_id: &str) -> bool {
        if let Some(job) = self.active_jobs.get(job_id) {
            job.cancel_token.cancel();
            true
        } else {
            false
        }
    }

    /// Remove a job from active tracking
    pub fn remove_job(&mut self, job_id: &str) {
        self.active_jobs.remove(job_id);
    }

    /// List active jobs in memory
    pub fn list_active_jobs(&self) -> Vec<(&str, &str)> {
        self.active_jobs
            .values()
            .map(|j| (j.job_id.as_str(), j.file_name.as_str()))
            .collect()
    }

    /// Check if a job is cancelled
    pub fn is_cancelled(&self, job_id: &str) -> bool {
        self.active_jobs
            .get(job_id)
            .map(|j| j.cancel_token.is_cancelled())
            .unwrap_or(false)
    }
}

impl Default for JobManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Database Operations
// ============================================================================

/// Escape a string for SQL
fn sanitize_sql_string(input: &str) -> String {
    input.replace('\'', "''")
}

/// Create the DOCUMENT_JOBS table if it doesn't exist
pub async fn ensure_jobs_table(engine: &SnowflakeEngine) -> Result<(), String> {
    let user = engine.get_user().await?;
    engine.setup_user_resources(&user).await?;
    let schema = engine.user_schema_name();

    let query = format!(
        r#"
        CREATE TABLE IF NOT EXISTS {schema}.DOCUMENT_JOBS (
            JOB_ID STRING PRIMARY KEY,
            FILE_NAME STRING NOT NULL,
            STAGE_PATH STRING NOT NULL,
            STATUS STRING NOT NULL DEFAULT 'pending',
            PROGRESS STRING,
            ERROR_MESSAGE STRING,
            CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
            UPDATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
        )
        "#
    );

    engine.execute_statement(&query).await.map(|_| ())
}

/// Insert a new job record
pub async fn insert_job(
    engine: &SnowflakeEngine,
    job_id: &str,
    file_name: &str,
    stage_path: &str,
) -> Result<(), String> {
    let schema = engine.user_schema_name();
    let escaped_job_id = sanitize_sql_string(job_id);
    let escaped_file_name = sanitize_sql_string(file_name);
    let escaped_stage_path = sanitize_sql_string(stage_path);

    let query = format!(
        r#"
        INSERT INTO {schema}.DOCUMENT_JOBS (JOB_ID, FILE_NAME, STAGE_PATH, STATUS)
        VALUES ('{escaped_job_id}', '{escaped_file_name}', '{escaped_stage_path}', 'pending')
        "#
    );

    engine.execute_statement(&query).await.map(|_| ())
}

/// Update job status
pub async fn update_job_status(
    engine: &SnowflakeEngine,
    job_id: &str,
    status: JobStatus,
    progress: Option<&str>,
    error_message: Option<&str>,
) -> Result<(), String> {
    let schema = engine.user_schema_name();
    let escaped_job_id = sanitize_sql_string(job_id);

    let progress_sql = progress
        .map(|p| format!("PROGRESS = '{}',", sanitize_sql_string(p)))
        .unwrap_or_default();

    let error_sql = error_message
        .map(|e| format!("ERROR_MESSAGE = '{}',", sanitize_sql_string(e)))
        .unwrap_or_default();

    let query = format!(
        r#"
        UPDATE {schema}.DOCUMENT_JOBS
        SET STATUS = '{status}',
            {progress_sql}
            {error_sql}
            UPDATED_AT = CURRENT_TIMESTAMP()
        WHERE JOB_ID = '{escaped_job_id}'
        "#
    );

    engine.execute_statement(&query).await.map(|_| ())
}

/// Get a single job by ID
pub async fn get_job(
    engine: &SnowflakeEngine,
    job_id: &str,
) -> Result<Option<DocumentJob>, String> {
    let schema = engine.user_schema_name();
    let escaped_job_id = sanitize_sql_string(job_id);

    let query = format!(
        r#"
        SELECT JOB_ID, FILE_NAME, STAGE_PATH, STATUS, PROGRESS, ERROR_MESSAGE,
               TO_VARCHAR(CREATED_AT) AS CREATED_AT, TO_VARCHAR(UPDATED_AT) AS UPDATED_AT
        FROM {schema}.DOCUMENT_JOBS
        WHERE JOB_ID = '{escaped_job_id}'
        "#
    );

    let result = engine.execute_query(&query, None, None, None).await?;

    if let Some(arr) = result.as_array() {
        if let Some(row) = arr.first() {
            return Ok(Some(parse_job_row(row)));
        }
    }

    Ok(None)
}

/// List all jobs (optionally filtered by status)
pub async fn list_jobs(
    engine: &SnowflakeEngine,
    status_filter: Option<JobStatus>,
    limit: Option<i32>,
) -> Result<Vec<DocumentJob>, String> {
    let schema = engine.user_schema_name();

    let status_clause = status_filter
        .map(|s| format!("WHERE STATUS = '{}'", s))
        .unwrap_or_default();

    let limit_clause = limit
        .map(|l| format!("LIMIT {}", l))
        .unwrap_or_else(|| "LIMIT 50".to_string());

    let query = format!(
        r#"
        SELECT JOB_ID, FILE_NAME, STAGE_PATH, STATUS, PROGRESS, ERROR_MESSAGE,
               TO_VARCHAR(CREATED_AT) AS CREATED_AT, TO_VARCHAR(UPDATED_AT) AS UPDATED_AT
        FROM {schema}.DOCUMENT_JOBS
        {status_clause}
        ORDER BY CREATED_AT DESC
        {limit_clause}
        "#
    );

    let result = engine.execute_query(&query, None, None, None).await?;

    let mut jobs = Vec::new();
    if let Some(arr) = result.as_array() {
        for row in arr {
            jobs.push(parse_job_row(row));
        }
    }

    Ok(jobs)
}

/// Parse a job row from JSON
fn parse_job_row(row: &Value) -> DocumentJob {
    let get_str = |key: &str| -> Option<String> {
        row.get(key)
            .or_else(|| row.get(&key.to_uppercase()))
            .or_else(|| row.get(&key.to_lowercase()))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    };

    DocumentJob {
        job_id: get_str("JOB_ID").unwrap_or_default(),
        file_name: get_str("FILE_NAME").unwrap_or_default(),
        stage_path: get_str("STAGE_PATH").unwrap_or_default(),
        status: get_str("STATUS")
            .map(|s| JobStatus::from_str(&s))
            .unwrap_or(JobStatus::Pending),
        progress: get_str("PROGRESS"),
        error_message: get_str("ERROR_MESSAGE"),
        created_at: get_str("CREATED_AT"),
        updated_at: get_str("UPDATED_AT"),
    }
}

/// Delete completed/failed jobs older than N days
pub async fn cleanup_old_jobs(engine: &SnowflakeEngine, days: i32) -> Result<i64, String> {
    let schema = engine.user_schema_name();

    let query = format!(
        r#"
        DELETE FROM {schema}.DOCUMENT_JOBS
        WHERE STATUS IN ('complete', 'failed', 'cancelled')
          AND CREATED_AT < DATEADD(day, -{days}, CURRENT_TIMESTAMP())
        "#
    );

    let result = engine.execute_statement(&query).await?;

    // Try to extract rows affected
    if let Some(affected) = result
        .get("number of rows deleted")
        .and_then(|v| v.as_i64())
    {
        Ok(affected)
    } else if let Some(affected) = result.get("rows_deleted").and_then(|v| v.as_i64()) {
        Ok(affected)
    } else {
        Ok(0)
    }
}

// ============================================================================
// CLI Integration
// ============================================================================

/// Print jobs list to console
pub fn print_jobs(jobs: &[DocumentJob], emitter: &EventEmitter) {
    if jobs.is_empty() {
        println!("\nNo document jobs found.");
        return;
    }

    println!("\n📋 Document Jobs:");
    println!(
        "{:<36} {:<30} {:<15} {:<20}",
        "JOB ID", "FILE NAME", "STATUS", "PROGRESS"
    );
    println!("{}", "-".repeat(101));

    for job in jobs {
        let status_icon = match job.status {
            JobStatus::Complete => "✓",
            JobStatus::Failed => "✗",
            JobStatus::Cancelled => "⊘",
            JobStatus::Processing | JobStatus::EmbeddingText | JobStatus::EmbeddingImages => "⋯",
            JobStatus::Pending => "○",
        };

        let progress = job.progress.as_deref().unwrap_or("-");
        let file_name_display = if job.file_name.len() > 28 {
            format!("{}...", &job.file_name[..25])
        } else {
            job.file_name.clone()
        };

        println!(
            "{} {:<34} {:<30} {:<15} {:<20}",
            status_icon,
            &job.job_id[..std::cmp::min(34, job.job_id.len())],
            file_name_display,
            job.status.to_string(),
            progress
        );

        // Emit job status event for JSON mode
        emitter.job_status(
            &job.job_id,
            &job.file_name,
            &job.status.to_string(),
            job.progress.as_deref(),
            job.error_message.as_deref(),
        );
    }
    println!();
}

/// Handle /jobs slash command
pub async fn handle_jobs_command(
    engine: &SnowflakeEngine,
    args: &str,
    emitter: &EventEmitter,
) -> Result<(), String> {
    // Ensure table exists
    ensure_jobs_table(engine).await?;

    let args_lower = args.trim().to_lowercase();

    if args_lower == "--clear" || args_lower == "clear" {
        // Clear old jobs
        let deleted = cleanup_old_jobs(engine, 30).await?;
        println!("Cleared {} old job records.", deleted);
        return Ok(());
    }

    // Parse status filter
    let status_filter = if args_lower.contains("pending") {
        Some(JobStatus::Pending)
    } else if args_lower.contains("processing") {
        Some(JobStatus::Processing)
    } else if args_lower.contains("complete") {
        Some(JobStatus::Complete)
    } else if args_lower.contains("failed") {
        Some(JobStatus::Failed)
    } else if args_lower.contains("cancelled") {
        Some(JobStatus::Cancelled)
    } else {
        None
    };

    let jobs = list_jobs(engine, status_filter, Some(50)).await?;
    print_jobs(&jobs, emitter);

    Ok(())
}

/// Handle /cancel slash command
pub async fn handle_cancel_command(
    engine: &SnowflakeEngine,
    job_manager: &mut JobManager,
    job_id: &str,
    emitter: &EventEmitter,
) -> Result<(), String> {
    let job_id = job_id.trim();

    if job_id.is_empty() {
        return Err("Usage: /cancel <job_id>".to_string());
    }

    // Try to cancel in-memory first
    let in_memory_cancelled = job_manager.cancel_job(job_id);

    // Update database status
    update_job_status(
        engine,
        job_id,
        JobStatus::Cancelled,
        None,
        Some("Cancelled by user"),
    )
    .await?;

    if in_memory_cancelled {
        println!("Job {} cancelled.", job_id);
    } else {
        println!(
            "Job {} marked as cancelled (was not actively running in this session).",
            job_id
        );
    }

    emitter.cancelled(job_id, Some("Cancelled by user"));

    Ok(())
}

/// Handle /timeout slash command
pub fn handle_timeout_command(job_manager: &mut JobManager, args: &str) -> Result<(), String> {
    let args = args.trim();

    if args.is_empty() {
        println!("Current timeout: {} seconds", job_manager.get_timeout());
        return Ok(());
    }

    match args.parse::<u64>() {
        Ok(seconds) => {
            job_manager.set_timeout(seconds);
            println!("Timeout set to {} seconds.", seconds);
            Ok(())
        }
        Err(_) => Err(format!("Invalid timeout value: {}", args)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_status_display() {
        assert_eq!(JobStatus::Pending.to_string(), "pending");
        assert_eq!(JobStatus::Complete.to_string(), "complete");
    }

    #[test]
    fn test_job_status_from_str() {
        assert_eq!(JobStatus::from_str("pending"), JobStatus::Pending);
        assert_eq!(JobStatus::from_str("COMPLETE"), JobStatus::Complete);
        assert_eq!(JobStatus::from_str("unknown"), JobStatus::Pending);
    }

    #[test]
    fn test_job_manager() {
        let mut manager = JobManager::new();

        let (job_id, _token) = manager.create_job("test.pdf");
        assert!(!manager.is_cancelled(&job_id));

        assert!(manager.cancel_job(&job_id));
        assert!(manager.is_cancelled(&job_id));
    }
}
