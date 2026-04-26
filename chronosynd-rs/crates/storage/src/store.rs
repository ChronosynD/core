//! BaselineStore, the SQLite-backed baseline persistence layer

use std::path::Path;

use rusqlite::{params, Connection};

use crate::audit::{self, AuditVerification};
use crate::error::{Result, StorageError};
use crate::schema::SCHEMA;
use crate::types::{MaintenanceWindow, StoredBaseline};

/// Tamper-evident store for per-process baselines and maintenance windows
pub struct BaselineStore {
    conn: Connection,
}

impl BaselineStore {
    /// Open or create a baseline store at `path`, applying the schema if needed
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path.as_ref())?;
        Self::with_connection(conn)
    }

    /// Open an in-memory baseline store for tests and ephemeral use
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::with_connection(conn)
    }

    fn with_connection(conn: Connection) -> Result<Self> {
        // WAL gives atomic writes via the write-ahead log file rather than
        // the rollback journal. A crash mid-write leaves the committed
        // prefix recoverable instead of risking a half-written page.
        // journal_mode is per-database and persists across opens
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    /// Insert a fresh baseline or replace the existing one for the same process
    pub fn put_baseline(&mut self, baseline: &StoredBaseline) -> Result<()> {
        if baseline.mean.len() != baseline.feature_dim {
            return Err(StorageError::InvalidArgument(format!(
                "mean length {} does not match feature_dim {}",
                baseline.mean.len(),
                baseline.feature_dim,
            )));
        }
        if baseline.std.len() != baseline.feature_dim {
            return Err(StorageError::InvalidArgument(format!(
                "std length {} does not match feature_dim {}",
                baseline.std.len(),
                baseline.feature_dim,
            )));
        }

        let tx = self.conn.transaction()?;
        let mean_json = serde_json::to_string(&baseline.mean)?;
        let std_json = serde_json::to_string(&baseline.std)?;
        tx.execute(
            "INSERT INTO baselines
                (process_key, feature_dim, mean_json, std_json,
                 estimator_kind, fitted_at_ns, sample_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(process_key) DO UPDATE SET
                feature_dim = excluded.feature_dim,
                mean_json = excluded.mean_json,
                std_json = excluded.std_json,
                estimator_kind = excluded.estimator_kind,
                fitted_at_ns = excluded.fitted_at_ns,
                sample_count = excluded.sample_count",
            params![
                baseline.process_key,
                baseline.feature_dim as i64,
                mean_json,
                std_json,
                baseline.estimator_kind,
                baseline.fitted_at_ns as i64,
                baseline.sample_count as i64,
            ],
        )?;
        let payload = serde_json::to_string(baseline)?;
        audit::append(&tx, baseline.fitted_at_ns, "baseline_put", &payload)?;
        tx.commit()?;
        Ok(())
    }

    /// Look up a baseline by process key
    pub fn get_baseline(&self, process_key: &str) -> Result<Option<StoredBaseline>> {
        let mut stmt = self.conn.prepare(
            "SELECT process_key, feature_dim, mean_json, std_json,
                    estimator_kind, fitted_at_ns, sample_count
             FROM baselines WHERE process_key = ?1",
        )?;
        let mut rows = stmt.query(params![process_key])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(decode_baseline_row(row)?));
        }
        Ok(None)
    }

    /// List every baseline currently in the store
    pub fn list_baselines(&self) -> Result<Vec<StoredBaseline>> {
        let mut stmt = self.conn.prepare(
            "SELECT process_key, feature_dim, mean_json, std_json,
                    estimator_kind, fitted_at_ns, sample_count
             FROM baselines ORDER BY process_key ASC",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(decode_baseline_row(row)?);
        }
        Ok(out)
    }

    /// Open a maintenance window starting at `start_ns`, returning its id
    pub fn start_maintenance_window(
        &mut self,
        start_ns: u64,
        note: Option<&str>,
    ) -> Result<i64> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT INTO maintenance_windows (start_ns, end_ns, note)
             VALUES (?1, NULL, ?2)",
            params![start_ns as i64, note],
        )?;
        let id = tx.last_insert_rowid();
        let payload = serde_json::to_string(&serde_json::json!({
            "id": id,
            "start_ns": start_ns,
            "note": note,
        }))?;
        audit::append(&tx, start_ns, "maintenance_start", &payload)?;
        tx.commit()?;
        Ok(id)
    }

    /// Close a previously-opened maintenance window, the window must be currently open
    pub fn end_maintenance_window(&mut self, id: i64, end_ns: u64) -> Result<()> {
        let tx = self.conn.transaction()?;
        let updated = tx.execute(
            "UPDATE maintenance_windows
             SET end_ns = ?1
             WHERE id = ?2 AND end_ns IS NULL",
            params![end_ns as i64, id],
        )?;
        if updated == 0 {
            return Err(StorageError::NotFound(format!(
                "no open maintenance window with id {id}"
            )));
        }
        let payload = serde_json::to_string(&serde_json::json!({
            "id": id,
            "end_ns": end_ns,
        }))?;
        audit::append(&tx, end_ns, "maintenance_end", &payload)?;
        tx.commit()?;
        Ok(())
    }

    /// Return the currently-open maintenance window, if any
    pub fn current_maintenance_window(&self) -> Result<Option<MaintenanceWindow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, start_ns, end_ns, note
             FROM maintenance_windows
             WHERE end_ns IS NULL
             ORDER BY id DESC LIMIT 1",
        )?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            return Ok(Some(decode_window_row(row)?));
        }
        Ok(None)
    }

    /// Walk the audit log start to end and report any tampering
    pub fn verify_audit_log(&self) -> Result<AuditVerification> {
        audit::verify(&self.conn)
    }

    /// Direct read of the connection. Gated behind the `test-utils` feature
    /// because direct SQL access bypasses the audit chain. Used only by the
    /// in-tree integration tests that inject tamper to verify detection
    #[cfg(any(test, feature = "test-utils"))]
    #[doc(hidden)]
    pub fn raw_connection(&self) -> &Connection {
        &self.conn
    }
}

fn decode_baseline_row(row: &rusqlite::Row<'_>) -> Result<StoredBaseline> {
    let process_key: String = row.get(0)?;
    let feature_dim: i64 = row.get(1)?;
    let mean_json: String = row.get(2)?;
    let std_json: String = row.get(3)?;
    let estimator_kind: String = row.get(4)?;
    let fitted_at_ns: i64 = row.get(5)?;
    let sample_count: i64 = row.get(6)?;

    Ok(StoredBaseline {
        process_key,
        feature_dim: feature_dim as usize,
        mean: serde_json::from_str(&mean_json)?,
        std: serde_json::from_str(&std_json)?,
        estimator_kind,
        fitted_at_ns: fitted_at_ns as u64,
        sample_count: sample_count as u64,
    })
}

fn decode_window_row(row: &rusqlite::Row<'_>) -> Result<MaintenanceWindow> {
    let id: i64 = row.get(0)?;
    let start_ns: i64 = row.get(1)?;
    let end_ns: Option<i64> = row.get(2)?;
    let note: Option<String> = row.get(3)?;
    Ok(MaintenanceWindow {
        id,
        start_ns: start_ns as u64,
        end_ns: end_ns.map(|v| v as u64),
        note,
    })
}
