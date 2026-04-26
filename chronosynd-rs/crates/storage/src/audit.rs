//! Hash-chained audit log, every state-changing operation appends a row
//! whose hash covers prev_hash plus operation plus payload plus timestamp
//! plus seq, anchored at the all-zero hash for the first row

use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};

use crate::error::Result;

/// Sentinel `prev_hash` for the very first audit row, all zeros
pub(crate) const ANCHOR_HASH: [u8; 32] = [0; 32];

/// Result of walking the audit log start to end and recomputing every hash
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditVerification {
    /// Whether every row's hash matched the recomputed value
    pub valid: bool,
    /// Sequence number of the highest row in the log, zero if empty
    pub last_seq: i64,
    /// First row where verification failed, `None` for clean logs
    pub broken_at_seq: Option<i64>,
    /// How many rows the log contained
    pub row_count: u64,
}

pub(crate) fn append(
    conn: &Connection,
    ts_ns: u64,
    operation: &str,
    payload: &str,
) -> Result<i64> {
    let prev_hash = latest_hash(conn)?;
    let next_seq = next_sequence(conn)?;
    let row_hash = compute_row_hash(next_seq, ts_ns, operation, payload, &prev_hash);

    conn.execute(
        "INSERT INTO audit_log (seq, ts_ns, operation, payload, prev_hash, hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            next_seq,
            ts_ns as i64,
            operation,
            payload,
            hex::encode(prev_hash),
            hex::encode(row_hash),
        ],
    )?;
    Ok(next_seq)
}

pub(crate) fn verify(conn: &Connection) -> Result<AuditVerification> {
    let mut stmt = conn.prepare(
        "SELECT seq, ts_ns, operation, payload, prev_hash, hash
         FROM audit_log ORDER BY seq ASC",
    )?;

    let mut rows = stmt.query([])?;
    let mut row_count: u64 = 0;
    let mut last_seq: i64 = 0;
    let mut prev_hash = ANCHOR_HASH;

    while let Some(row) = rows.next()? {
        let seq: i64 = row.get(0)?;
        let ts_ns: i64 = row.get(1)?;
        let operation: String = row.get(2)?;
        let payload: String = row.get(3)?;
        let stored_prev_hex: String = row.get(4)?;
        let stored_hash_hex: String = row.get(5)?;

        let stored_prev = decode_hash(&stored_prev_hex)?;
        let stored_hash = decode_hash(&stored_hash_hex)?;

        if stored_prev != prev_hash {
            return Ok(AuditVerification {
                valid: false,
                last_seq,
                broken_at_seq: Some(seq),
                row_count,
            });
        }

        let recomputed = compute_row_hash(seq, ts_ns as u64, &operation, &payload, &prev_hash);
        if recomputed != stored_hash {
            return Ok(AuditVerification {
                valid: false,
                last_seq,
                broken_at_seq: Some(seq),
                row_count,
            });
        }

        prev_hash = stored_hash;
        last_seq = seq;
        row_count += 1;
    }

    Ok(AuditVerification {
        valid: true,
        last_seq,
        broken_at_seq: None,
        row_count,
    })
}

fn latest_hash(conn: &Connection) -> Result<[u8; 32]> {
    let value: Option<String> = conn
        .query_row(
            "SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    match value {
        None => Ok(ANCHOR_HASH),
        Some(hex_string) => decode_hash(&hex_string),
    }
}

fn next_sequence(conn: &Connection) -> Result<i64> {
    let value: Option<i64> = conn
        .query_row("SELECT MAX(seq) FROM audit_log", [], |row| row.get(0))
        .optional()?
        .flatten();
    Ok(value.unwrap_or(0) + 1)
}

fn compute_row_hash(
    seq: i64,
    ts_ns: u64,
    operation: &str,
    payload: &str,
    prev_hash: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(seq.to_le_bytes());
    hasher.update(ts_ns.to_le_bytes());
    hasher.update((operation.len() as u64).to_le_bytes());
    hasher.update(operation.as_bytes());
    hasher.update((payload.len() as u64).to_le_bytes());
    hasher.update(payload.as_bytes());
    hasher.update(prev_hash);
    hasher.finalize().into()
}

fn decode_hash(hex_string: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(hex_string).map_err(|err| {
        crate::error::StorageError::InvalidArgument(format!("audit log hash not hex: {err}"))
    })?;
    let array: [u8; 32] = bytes.try_into().map_err(|_| {
        crate::error::StorageError::InvalidArgument(
            "audit log hash had unexpected byte length".into(),
        )
    })?;
    Ok(array)
}
