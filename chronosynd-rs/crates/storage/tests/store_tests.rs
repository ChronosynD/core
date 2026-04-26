//! Unit tests for the BaselineStore

use chronosynd_storage::{BaselineStore, StorageError, StoredBaseline};

fn sample_baseline(key: &str) -> StoredBaseline {
    StoredBaseline {
        process_key: key.into(),
        feature_dim: 4,
        mean: vec![0.1, 0.2, 0.3, 0.4],
        std: vec![1.0, 1.1, 1.2, 1.3],
        estimator_kind: "sediment_trim30".into(),
        fitted_at_ns: 1_700_000_000_000,
        sample_count: 500,
    }
}

#[test]
fn put_and_get_round_trip() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    let baseline = sample_baseline("image:/usr/sbin/nginx");
    store.put_baseline(&baseline).unwrap();

    let fetched = store.get_baseline(&baseline.process_key).unwrap().unwrap();
    assert_eq!(fetched, baseline);
}

#[test]
fn get_returns_none_for_missing_key() {
    let store = BaselineStore::open_in_memory().unwrap();
    let fetched = store.get_baseline("image:/nope").unwrap();
    assert!(fetched.is_none());
}

#[test]
fn put_replaces_existing_baseline_for_same_key() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    let mut baseline = sample_baseline("k");
    store.put_baseline(&baseline).unwrap();
    baseline.sample_count = 600;
    baseline.mean[0] = 9.9;
    store.put_baseline(&baseline).unwrap();

    let fetched = store.get_baseline("k").unwrap().unwrap();
    assert_eq!(fetched.sample_count, 600);
    assert_eq!(fetched.mean[0], 9.9);
}

#[test]
fn list_returns_baselines_sorted_by_key() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    store.put_baseline(&sample_baseline("c")).unwrap();
    store.put_baseline(&sample_baseline("a")).unwrap();
    store.put_baseline(&sample_baseline("b")).unwrap();

    let list = store.list_baselines().unwrap();
    let keys: Vec<&str> = list.iter().map(|b| b.process_key.as_str()).collect();
    assert_eq!(keys, vec!["a", "b", "c"]);
}

#[test]
fn put_rejects_mean_dimension_mismatch() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    let mut baseline = sample_baseline("k");
    baseline.mean.pop();
    let err = store.put_baseline(&baseline).unwrap_err();
    assert!(matches!(err, StorageError::InvalidArgument(_)));
}

#[test]
fn put_rejects_std_dimension_mismatch() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    let mut baseline = sample_baseline("k");
    baseline.std.push(99.0);
    let err = store.put_baseline(&baseline).unwrap_err();
    assert!(matches!(err, StorageError::InvalidArgument(_)));
}

#[test]
fn maintenance_window_lifecycle() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    let id = store
        .start_maintenance_window(1_000, Some("kernel upgrade"))
        .unwrap();

    let current = store.current_maintenance_window().unwrap().unwrap();
    assert_eq!(current.id, id);
    assert_eq!(current.start_ns, 1_000);
    assert_eq!(current.end_ns, None);
    assert_eq!(current.note.as_deref(), Some("kernel upgrade"));

    store.end_maintenance_window(id, 2_000).unwrap();
    assert!(store.current_maintenance_window().unwrap().is_none());
}

#[test]
fn end_maintenance_window_rejects_unknown_id() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    let err = store.end_maintenance_window(999, 1).unwrap_err();
    assert!(matches!(err, StorageError::NotFound(_)));
}

#[test]
fn end_maintenance_window_rejects_already_closed_window() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    let id = store.start_maintenance_window(100, None).unwrap();
    store.end_maintenance_window(id, 200).unwrap();
    let err = store.end_maintenance_window(id, 300).unwrap_err();
    assert!(matches!(err, StorageError::NotFound(_)));
}

#[test]
fn empty_store_audit_log_verifies_clean() {
    let store = BaselineStore::open_in_memory().unwrap();
    let report = store.verify_audit_log().unwrap();
    assert!(report.valid);
    assert_eq!(report.row_count, 0);
    assert_eq!(report.broken_at_seq, None);
}

#[test]
fn populated_audit_log_verifies_clean() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    store.put_baseline(&sample_baseline("a")).unwrap();
    store.put_baseline(&sample_baseline("b")).unwrap();
    let id = store.start_maintenance_window(10, Some("test")).unwrap();
    store.end_maintenance_window(id, 20).unwrap();

    let report = store.verify_audit_log().unwrap();
    assert!(report.valid, "audit verification failed: {report:?}");
    assert_eq!(report.row_count, 4);
    assert!(report.last_seq >= 4);
}

#[test]
fn tampering_with_a_payload_is_detected() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    store.put_baseline(&sample_baseline("a")).unwrap();
    store.put_baseline(&sample_baseline("b")).unwrap();

    store
        .raw_connection()
        .execute(
            "UPDATE audit_log SET payload = '{\"tampered\": true}' WHERE seq = 1",
            [],
        )
        .unwrap();

    let report = store.verify_audit_log().unwrap();
    assert!(!report.valid);
    assert_eq!(report.broken_at_seq, Some(1));
}

#[test]
fn tampering_with_a_hash_is_detected() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    store.put_baseline(&sample_baseline("a")).unwrap();
    store.put_baseline(&sample_baseline("b")).unwrap();

    store
        .raw_connection()
        .execute(
            "UPDATE audit_log SET hash = ?1 WHERE seq = 2",
            rusqlite::params!["00".repeat(32)],
        )
        .unwrap();

    let report = store.verify_audit_log().unwrap();
    assert!(!report.valid);
    assert_eq!(report.broken_at_seq, Some(2));
}

#[test]
fn deleting_a_row_breaks_the_chain() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    store.put_baseline(&sample_baseline("a")).unwrap();
    store.put_baseline(&sample_baseline("b")).unwrap();
    store.put_baseline(&sample_baseline("c")).unwrap();

    store
        .raw_connection()
        .execute("DELETE FROM audit_log WHERE seq = 2", [])
        .unwrap();

    let report = store.verify_audit_log().unwrap();
    assert!(!report.valid);
    assert_eq!(report.broken_at_seq, Some(3));
}
