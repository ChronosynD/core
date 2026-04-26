//! SQLite schema for the baseline store, applied via `CREATE ... IF NOT
//! EXISTS` on every open, the audit log holds one row per state-changing
//! operation and forms the hash chain the verify pass walks

pub const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS baselines (
    process_key    TEXT PRIMARY KEY NOT NULL,
    feature_dim    INTEGER NOT NULL,
    mean_json      TEXT NOT NULL,
    std_json       TEXT NOT NULL,
    estimator_kind TEXT NOT NULL,
    fitted_at_ns   INTEGER NOT NULL,
    sample_count   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS maintenance_windows (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ns  INTEGER NOT NULL,
    end_ns    INTEGER,
    note      TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ns      INTEGER NOT NULL,
    operation  TEXT NOT NULL,
    payload    TEXT NOT NULL,
    prev_hash  TEXT NOT NULL,
    hash       TEXT NOT NULL
);
"#;
