use rusqlite::params;
use serde::{Deserialize, Serialize};
use tokio_rusqlite::Connection;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ColumnInfo {
    pub column_name: String,
    pub ordinal_position: i64,
    pub column_default: Option<String>,
    pub is_nullable: String,
    pub data_type: String,
}

pub struct CacheConnection {
    conn: Option<Connection>,
}

impl CacheConnection {
    pub async fn new(path: Option<&str>, key: &str) -> Result<Self, tokio_rusqlite::Error> {
        let conn = match path {
            Some(p) => Connection::open(p).await?,
            None => Connection::open_in_memory().await?,
        };

        let key = key.to_string();
        conn.call(move |conn| {
            conn.pragma_update(None, "key", &key)?;
            conn.pragma_update(None, "journal_mode", "WAL")?;
            conn.pragma_update(None, "busy_timeout", 5000)?;
            Ok(())
        })
        .await?;

        let cache_conn = Self { conn: Some(conn) };
        cache_conn.create_tables().await?;
        Ok(cache_conn)
    }

    async fn create_tables(&self) -> Result<(), tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(|conn| {
                conn.execute_batch(
                    "
                CREATE TABLE IF NOT EXISTS schemas (
                    database TEXT NOT NULL,
                    name TEXT NOT NULL,
                    PRIMARY KEY (database, name)
                );
                CREATE TABLE IF NOT EXISTS warehouses (
                    name TEXT NOT NULL PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS databases (
                    name TEXT NOT NULL PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS tables (
                    database TEXT NOT NULL,
                    schema TEXT NOT NULL,
                    name TEXT NOT NULL,
                    PRIMARY KEY (database, schema, name)
                );
                CREATE TABLE IF NOT EXISTS stages (
                    name TEXT NOT NULL PRIMARY KEY
                );
                CREATE TABLE IF NOT EXISTS stage_files (
                    stage_name TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    PRIMARY KEY (stage_name, file_name)
                );
                CREATE TABLE IF NOT EXISTS table_schemas (
                    database TEXT NOT NULL,
                    schema TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    columns_json TEXT NOT NULL,
                    PRIMARY KEY (database, schema, table_name)
                );
                ",
                )?;
                Ok(())
            })
            .await
    }


    pub async fn get_warehouses(&self) -> Result<Vec<String>, tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(|conn| {
                let mut stmt = conn.prepare("SELECT name FROM warehouses")?;
                let mut rows = stmt.query([])?;
                let mut result = Vec::new();
                while let Some(row) = rows.next()? {
                    result.push(row.get(0)?);
                }
                Ok(result)
            })
            .await
    }

    pub async fn set_warehouses(
        &self,
        warehouses: Vec<String>,
    ) -> Result<(), tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                let tx = conn.transaction()?;
                tx.execute("DELETE FROM warehouses", [])?;
                let mut stmt = tx.prepare("INSERT INTO warehouses (name) VALUES (?)")?;
                for wh in warehouses {
                    stmt.execute([wh])?;
                }
                drop(stmt);
                tx.commit()?;
                Ok(())
            })
            .await
    }

    pub async fn get_databases(&self) -> Result<Vec<String>, tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(|conn| {
                let mut stmt = conn.prepare("SELECT name FROM databases")?;
                let mut rows = stmt.query([])?;
                let mut result = Vec::new();
                while let Some(row) = rows.next()? {
                    result.push(row.get(0)?);
                }
                Ok(result)
            })
            .await
    }

    pub async fn set_databases(&self, databases: Vec<String>) -> Result<(), tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                let tx = conn.transaction()?;
                tx.execute("DELETE FROM databases", [])?;
                let mut stmt = tx.prepare("INSERT INTO databases (name) VALUES (?)")?;
                for db in databases {
                    stmt.execute([db])?;
                }
                drop(stmt);
                tx.commit()?;
                Ok(())
            })
            .await
    }

    pub async fn get_schemas(&self, database: &str) -> Result<Vec<String>, tokio_rusqlite::Error> {
        let database = database.to_string();
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                let mut stmt = conn.prepare("SELECT name FROM schemas WHERE database = ?")?;
                let mut rows = stmt.query([database])?;
                let mut result = Vec::new();
                while let Some(row) = rows.next()? {
                    result.push(row.get(0)?);
                }
                Ok(result)
            })
            .await
    }

    pub async fn set_schemas(
        &self,
        database: String,
        schemas: Vec<String>,
    ) -> Result<(), tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                let tx = conn.transaction()?;
                tx.execute("DELETE FROM schemas WHERE database = ?", [&database])?;
                let mut stmt = tx.prepare("INSERT INTO schemas (database, name) VALUES (?, ?)")?;
                for schema in schemas {
                    stmt.execute([&database, &schema])?;
                }
                drop(stmt);
                tx.commit()?;
                Ok(())
            })
            .await
    }

    pub async fn get_tables_in_schema(
        &self,
        database: &str,
        schema: &str,
    ) -> Result<Vec<String>, tokio_rusqlite::Error> {
        let database = database.to_string();
        let schema = schema.to_string();
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                let mut stmt =
                    conn.prepare("SELECT name FROM tables WHERE database = ? AND schema = ?")?;
                let mut rows = stmt.query([database, schema])?;
                let mut result = Vec::new();
                while let Some(row) = rows.next()? {
                    result.push(row.get(0)?);
                }
                Ok(result)
            })
            .await
    }

    pub async fn set_tables_in_schema(
        &self,
        database: String,
        schema: String,
        tables: Vec<String>,
    ) -> Result<(), tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                let tx = conn.transaction()?;
                tx.execute(
                    "DELETE FROM tables WHERE database = ? AND schema = ?",
                    [&database, &schema],
                )?;
                let mut stmt =
                    tx.prepare("INSERT INTO tables (database, schema, name) VALUES (?, ?, ?)")?;
                for table in tables {
                    stmt.execute([&database, &schema, &table])?;
                }
                drop(stmt);
                tx.commit()?;
                Ok(())
            })
            .await
    }

    pub async fn invalidate_tables_in_schema(
        &self,
        database: &str,
        schema: &str,
    ) -> Result<(), tokio_rusqlite::Error> {
        let database = database.to_string();
        let schema = schema.to_string();
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                conn.execute(
                    "DELETE FROM tables WHERE database = ? AND schema = ?",
                    [&database, &schema],
                )?;
                Ok(())
            })
            .await
    }

    pub async fn get_all_tables(
        &self,
    ) -> Result<Vec<(String, String, String)>, tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(|conn| {
                let mut stmt = conn.prepare("SELECT database, schema, name FROM tables")?;
                let mut rows = stmt.query([])?;
                let mut result = Vec::new();
                while let Some(row) = rows.next()? {
                    result.push((row.get(0)?, row.get(1)?, row.get(2)?));
                }
                Ok(result)
            })
            .await
    }

    pub async fn set_all_tables(
        &self,
        tables: Vec<(String, String, String)>,
    ) -> Result<(), tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                let tx = conn.transaction()?;
                tx.execute("DELETE FROM tables", [])?;
                let mut stmt =
                    tx.prepare("INSERT INTO tables (database, schema, name) VALUES (?, ?, ?)")?;
                for (db, schema, name) in tables {
                    stmt.execute([db, schema, name])?;
                }
                drop(stmt);
                tx.commit()?;
                Ok(())
            })
            .await
    }

    pub async fn get_stages(&self) -> Result<Vec<String>, tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(|conn| {
                let mut stmt = conn.prepare("SELECT name FROM stages")?;
                let mut rows = stmt.query([])?;
                let mut result = Vec::new();
                while let Some(row) = rows.next()? {
                    result.push(row.get(0)?);
                }
                Ok(result)
            })
            .await
    }

    pub async fn set_stages(&self, stages: Vec<String>) -> Result<(), tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                let tx = conn.transaction()?;
                tx.execute("DELETE FROM stages", [])?;
                let mut stmt = tx.prepare("INSERT INTO stages (name) VALUES (?)")?;
                for stage in stages {
                    stmt.execute([stage])?;
                }
                drop(stmt);
                tx.commit()?;
                Ok(())
            })
            .await
    }

    pub async fn invalidate_stages(&self) -> Result<(), tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                conn.execute("DELETE FROM stages", [])?;
                Ok(())
            })
            .await
    }

    pub async fn get_stage_files(
        &self,
        stage_name: &str,
    ) -> Result<Vec<String>, tokio_rusqlite::Error> {
        let stage_name = stage_name.to_string();
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                let mut stmt =
                    conn.prepare("SELECT file_name FROM stage_files WHERE stage_name = ?")?;
                let mut rows = stmt.query([stage_name])?;
                let mut result = Vec::new();
                while let Some(row) = rows.next()? {
                    result.push(row.get(0)?);
                }
                Ok(result)
            })
            .await
    }

    pub async fn set_stage_files(
        &self,
        stage_name: String,
        files: Vec<String>,
    ) -> Result<(), tokio_rusqlite::Error> {
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                let tx = conn.transaction()?;
                tx.execute(
                    "DELETE FROM stage_files WHERE stage_name = ?",
                    [&stage_name],
                )?;
                let mut stmt =
                    tx.prepare("INSERT INTO stage_files (stage_name, file_name) VALUES (?, ?)")?;
                for file in files {
                    stmt.execute([&stage_name, &file])?;
                }
                drop(stmt);
                tx.commit()?;
                Ok(())
            })
            .await
    }

    pub async fn invalidate_stage_files(
        &self,
        stage_name: &str,
    ) -> Result<(), tokio_rusqlite::Error> {
        let stage_name = stage_name.to_string();
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                conn.execute("DELETE FROM stage_files WHERE stage_name = ?", [&stage_name])?;
                Ok(())
            })
            .await
    }

    // Update get_table_schema
    pub async fn get_table_schema(
        &self,
        db: &str,
        schema: &str,
        table: &str,
    ) -> Result<Option<Vec<ColumnInfo>>, tokio_rusqlite::Error> {
        let db = db.to_string();
        let schema = schema.to_string();
        let table = table.to_string();
        self.conn.as_ref().unwrap().call(move |conn| {
            let mut stmt = conn.prepare("SELECT columns_json FROM table_schemas WHERE database = ? AND schema = ? AND table_name = ?")?;
            let mut rows = stmt.query([db, schema, table])?;
            if let Some(row) = rows.next()? {
                let json_str: String = row.get(0)?;
                let columns: Vec<ColumnInfo> = serde_json::from_str(&json_str)
                    .map_err(|e| rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e)))?;
                Ok(Some(columns))
            } else {
                Ok(None)
            }
        }).await
    }

    // Update set_table_schema to accept ColumnInfo
    pub async fn set_table_schema(
        &self,
        db: String,
        schema: String,
        table: String,
        columns: Vec<ColumnInfo>,
    ) -> Result<(), tokio_rusqlite::Error> {
        self.conn.as_ref().unwrap().call(move |conn| {
            let json_str = serde_json::to_string(&columns)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;
            conn.execute(
                "INSERT OR REPLACE INTO table_schemas (database, schema, table_name, columns_json) VALUES (?, ?, ?, ?)",
                [&db as &dyn rusqlite::ToSql, &schema, &table, &json_str],
            )?;
            Ok(())
        }).await
    }

    pub async fn set_conversation_data(
        &self,
        conversation_id: String,
        key: String,
        value: String,
    ) -> Result<(), tokio_rusqlite::Error> {
        self.conn.as_ref().unwrap().call(move |conn| {
            conn.execute(
                "INSERT OR REPLACE INTO conversation_data (conversation_id, key, value) VALUES (?, ?, ?)",
                [&conversation_id, &key, &value],
            )?;
            Ok(())
        }).await
    }

    pub async fn get_conversation_data(
        &self,
        conversation_id: &str,
        key: &str,
    ) -> Result<Option<String>, tokio_rusqlite::Error> {
        let conversation_id = conversation_id.to_string();
        let key = key.to_string();
        self.conn
            .as_ref()
            .unwrap()
            .call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT value FROM conversation_data WHERE conversation_id = ? AND key = ?",
                )?;
                let mut rows = stmt.query([conversation_id, key])?;
                if let Some(row) = rows.next()? {
                    Ok(Some(row.get(0)?))
                } else {
                    Ok(None)
                }
            })
            .await
    }

    /// Delete conversation-specific data
    pub async fn delete_conversation_data(
        &self,
        conversation_id: &str,
        key: &str,
    ) -> Result<(), String> {
        if let Some(conn) = &self.conn {
            let conversation_id = conversation_id.to_string();
            let key = key.to_string();

            conn.call(move |conn| {
                conn.execute(
                    "DELETE FROM conversation_data WHERE conversation_id = ?1 AND key = ?2",
                    params![conversation_id, key],
                )?;
                Ok(())
            })
            .await
            .map_err(|e| format!("Failed to delete conversation data: {}", e))
        } else {
            Err("Cache connection not initialized".to_string())
        }
    }

    /// List all conversation data for a conversation
    pub async fn list_conversation_data(
        &self,
        conversation_id: &str,
    ) -> Result<Vec<(String, String)>, String> {
        if let Some(conn) = &self.conn {
            let conversation_id = conversation_id.to_string();

            let result = conn.call(move |conn| {
                let mut stmt = conn.prepare(
                    "SELECT key, value FROM conversation_data WHERE conversation_id = ?1 ORDER BY key"
                )?;
                let rows = stmt.query_map(params![conversation_id], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?
                    ))
                })?;
                
                let mut results = Vec::new();
                for row in rows {
                    results.push(row?);
                }
                Ok(results)
            }).await;

            match result {
                Ok(data) => Ok(data),
                Err(e) => Err(format!("Failed to list conversation data: {}", e)),
            }
        } else {
            Err("Cache connection not initialized".to_string())
        }
    }

    /// Clear all messages for a conversation
    pub async fn clear_conversation(&self, conversation_id: &str) -> Result<(), String> {
        if let Some(conn) = &self.conn {
            let conversation_id = conversation_id.to_string();

            let result = conn
                .call(move |conn| {
                    conn.execute(
                        "DELETE FROM conversation_messages WHERE conversation_id = ?1",
                        params![conversation_id],
                    )?;
                    Ok(())
                })
                .await;

            match result {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("Failed to clear conversation: {}", e)),
            }
        } else {
            Err("Cache connection not initialized".to_string())
        }
    }

    pub async fn close(&mut self) -> Result<(), tokio_rusqlite::Error> {
        if let Some(conn) = self.conn.as_ref() {
            // Best effort checkpoint.
            let _ = conn
                .call(|c| {
                    c.pragma_update(None, "wal_checkpoint", "TRUNCATE")?;
                    Ok(())
                })
                .await;
        }

        if let Some(conn) = self.conn.take() {
            conn.close().await
        } else {
            Ok(())
        }
    }
}
