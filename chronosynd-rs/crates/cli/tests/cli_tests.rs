//! Integration tests for the operator CLI, each test creates a fresh
//! on-disk store in a unique temp directory and drives `run` with
//! captured stdout, no subprocesses so the tests stay cross-platform

use std::path::{Path, PathBuf};

use chronosynd_cli::{run, EXIT_ERR, EXIT_OK, EXIT_TAMPERED};
use chronosynd_storage::{BaselineStore, StoredBaseline};

struct TempStore {
    dir: PathBuf,
    path: PathBuf,
}

impl TempStore {
    fn new(label: &str) -> Self {
        let mut dir = std::env::temp_dir();
        let unique = format!(
            "chronosynd_cli_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        );
        dir.push(unique);
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("chronosynd.db");
        Self { dir, path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempStore {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn invoke(store: &Path, args: &[&str]) -> (i32, String) {
    let mut argv: Vec<String> = vec!["chronosynd".into()];
    argv.push("--store".into());
    argv.push(store.to_string_lossy().into_owned());
    argv.extend(args.iter().map(|s| s.to_string()));
    let mut buffer = Vec::new();
    let code = run(argv, &mut buffer);
    (code, String::from_utf8(buffer).expect("CLI output is utf-8"))
}

fn sample_baseline(key: &str) -> StoredBaseline {
    StoredBaseline {
        process_key: key.into(),
        feature_dim: 3,
        mean: vec![0.0, 1.0, 2.0],
        std: vec![1.0, 1.0, 1.0],
        estimator_kind: "sediment_trim30".into(),
        fitted_at_ns: 1_700_000_000_000,
        sample_count: 500,
    }
}

#[test]
fn baseline_list_on_empty_store_reports_nothing_recorded() {
    let temp = TempStore::new("list_empty");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let (code, out) = invoke(temp.path(), &["baseline", "list"]);
    assert_eq!(code, EXIT_OK);
    assert!(out.contains("no baselines recorded"), "got: {out}");
}

#[test]
fn baseline_list_shows_records_in_a_table() {
    let temp = TempStore::new("list_populated");
    {
        let mut store = BaselineStore::open(temp.path()).unwrap();
        store.put_baseline(&sample_baseline("a")).unwrap();
        store.put_baseline(&sample_baseline("b")).unwrap();
    }

    let (code, out) = invoke(temp.path(), &["baseline", "list"]);
    assert_eq!(code, EXIT_OK);
    assert!(out.contains("PROCESS_KEY"), "header missing: {out}");
    assert!(out.contains("a"));
    assert!(out.contains("b"));
    assert!(out.contains("sediment_trim30"));
}

#[test]
fn baseline_show_prints_full_record() {
    let temp = TempStore::new("show_record");
    {
        let mut store = BaselineStore::open(temp.path()).unwrap();
        store.put_baseline(&sample_baseline("nginx")).unwrap();
    }

    let (code, out) = invoke(temp.path(), &["baseline", "show", "nginx"]);
    assert_eq!(code, EXIT_OK);
    assert!(out.contains("process_key:"));
    assert!(out.contains("nginx"));
    assert!(out.contains("feature_dim:    3"));
    assert!(out.contains("sample_count:   500"));
}

#[test]
fn baseline_show_for_unknown_key_exits_with_error() {
    let temp = TempStore::new("show_missing");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let (code, _) = invoke(temp.path(), &["baseline", "show", "nope"]);
    assert_eq!(code, EXIT_ERR);
}

#[test]
fn maintenance_start_then_current_then_end_round_trip() {
    let temp = TempStore::new("maintenance_lifecycle");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let (start_code, start_out) =
        invoke(temp.path(), &["maintenance", "start", "--note", "kernel patch"]);
    assert_eq!(start_code, EXIT_OK);
    assert!(start_out.contains("started maintenance window 1"));

    let (current_code, current_out) = invoke(temp.path(), &["maintenance", "current"]);
    assert_eq!(current_code, EXIT_OK);
    assert!(current_out.contains("id:        1"));
    assert!(current_out.contains("note:      kernel patch"));
    assert!(current_out.contains("end_ns:    open"));

    let (end_code, end_out) = invoke(temp.path(), &["maintenance", "end", "1"]);
    assert_eq!(end_code, EXIT_OK);
    assert!(end_out.contains("ended maintenance window 1"));

    let (after_code, after_out) = invoke(temp.path(), &["maintenance", "current"]);
    assert_eq!(after_code, EXIT_OK);
    assert!(after_out.contains("no open maintenance window"));
}

#[test]
fn maintenance_end_unknown_id_exits_with_error() {
    let temp = TempStore::new("maintenance_unknown");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let (code, _) = invoke(temp.path(), &["maintenance", "end", "99"]);
    assert_eq!(code, EXIT_ERR);
}

#[test]
fn verify_store_on_clean_log_exits_zero() {
    let temp = TempStore::new("verify_clean");
    {
        let mut store = BaselineStore::open(temp.path()).unwrap();
        store.put_baseline(&sample_baseline("a")).unwrap();
        store.put_baseline(&sample_baseline("b")).unwrap();
    }

    let (code, out) = invoke(temp.path(), &["verify-store"]);
    assert_eq!(code, EXIT_OK);
    assert!(out.contains("status:  CLEAN"), "got: {out}");
    assert!(out.contains("rows:    2"), "got: {out}");
}

#[test]
fn verify_store_on_tampered_log_exits_with_tampered_code() {
    let temp = TempStore::new("verify_tampered");
    {
        let mut store = BaselineStore::open(temp.path()).unwrap();
        store.put_baseline(&sample_baseline("a")).unwrap();
        store.put_baseline(&sample_baseline("b")).unwrap();
        store
            .raw_connection()
            .execute(
                "UPDATE audit_log SET payload = '{\"hacked\": true}' WHERE seq = 1",
                [],
            )
            .unwrap();
    }

    let (code, out) = invoke(temp.path(), &["verify-store"]);
    assert_eq!(code, EXIT_TAMPERED);
    assert!(out.contains("status:  TAMPERED"));
    assert!(out.contains("seq 1"));
}

#[test]
fn unknown_subcommand_exits_with_error() {
    let temp = TempStore::new("unknown");
    let (code, _) = invoke(temp.path(), &["nope"]);
    assert_eq!(code, EXIT_ERR);
}

fn write_csv(dir: &Path, name: &str, rows: &[&[f64]]) -> PathBuf {
    let path = dir.join(name);
    let mut buffer = String::new();
    for row in rows {
        let cells: Vec<String> = row.iter().map(|v| format!("{v:.6}")).collect();
        buffer.push_str(&cells.join(","));
        buffer.push('\n');
    }
    std::fs::write(&path, buffer).expect("write csv fixture");
    path
}

#[test]
fn fit_baseline_with_sediment_persists_a_record_with_correct_metadata() {
    let temp = TempStore::new("fit_sediment");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let csv = write_csv(
        &temp.dir,
        "obs.csv",
        &[
            &[0.0, 0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[-1.0, -1.0, -1.0, -1.0],
            &[0.5, 0.5, 0.5, 0.5],
            &[-0.5, -0.5, -0.5, -0.5],
        ],
    );
    let (code, out) = invoke(
        temp.path(),
        &[
            "fit-baseline",
            "nginx",
            "--input",
            csv.to_str().unwrap(),
            "--estimator",
            "sediment",
            "--trim-fraction",
            "0.2",
        ],
    );
    assert_eq!(code, EXIT_OK, "got: {out}");
    assert!(out.contains("fit baseline for nginx"), "got: {out}");
    assert!(out.contains("5 samples"));
    assert!(out.contains("4 features"));
    assert!(out.contains("sediment_trim20"));

    let store = BaselineStore::open(temp.path()).unwrap();
    let record = store.get_baseline("nginx").unwrap().expect("baseline missing");
    assert_eq!(record.feature_dim, 4);
    assert_eq!(record.sample_count, 5);
    assert_eq!(record.estimator_kind, "sediment_trim20");
    assert_eq!(record.mean.len(), 4);
    assert_eq!(record.std.len(), 4);
}

#[test]
fn fit_baseline_with_naive_uses_naive_kind() {
    let temp = TempStore::new("fit_naive");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let csv = write_csv(
        &temp.dir,
        "obs.csv",
        &[&[0.0, 1.0], &[2.0, 3.0], &[4.0, 5.0]],
    );
    let (code, out) = invoke(
        temp.path(),
        &[
            "fit-baseline",
            "sshd",
            "--input",
            csv.to_str().unwrap(),
            "--estimator",
            "naive",
        ],
    );
    assert_eq!(code, EXIT_OK, "got: {out}");
    assert!(out.contains("naive"));

    let store = BaselineStore::open(temp.path()).unwrap();
    let record = store.get_baseline("sshd").unwrap().expect("baseline missing");
    assert_eq!(record.estimator_kind, "naive");
}

#[test]
fn fit_baseline_keeps_audit_log_clean() {
    let temp = TempStore::new("fit_audit");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let csv = write_csv(&temp.dir, "obs.csv", &[&[0.0, 0.0], &[1.0, 1.0], &[-1.0, -1.0]]);
    let (fit_code, _) = invoke(
        temp.path(),
        &[
            "fit-baseline",
            "proc",
            "--input",
            csv.to_str().unwrap(),
            "--estimator",
            "naive",
        ],
    );
    assert_eq!(fit_code, EXIT_OK);

    let (verify_code, verify_out) = invoke(temp.path(), &["verify-store"]);
    assert_eq!(verify_code, EXIT_OK, "got: {verify_out}");
    assert!(verify_out.contains("status:  CLEAN"));
}

#[test]
fn fit_baseline_rejects_inconsistent_row_widths() {
    let temp = TempStore::new("fit_jagged");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let csv = temp.dir.join("jagged.csv");
    std::fs::write(&csv, "1.0,2.0,3.0\n4.0,5.0\n").unwrap();

    let (code, _) = invoke(
        temp.path(),
        &[
            "fit-baseline",
            "p",
            "--input",
            csv.to_str().unwrap(),
            "--estimator",
            "naive",
        ],
    );
    assert_eq!(code, EXIT_ERR);
}

#[test]
fn fit_baseline_rejects_missing_input_file() {
    let temp = TempStore::new("fit_missing");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let (code, _) = invoke(
        temp.path(),
        &[
            "fit-baseline",
            "p",
            "--input",
            "definitely_not_there.csv",
            "--estimator",
            "naive",
        ],
    );
    assert_eq!(code, EXIT_ERR);
}

#[test]
fn run_with_seed_demo_warms_one_baseline_and_alerts_on_anomalous_window() {
    let temp = TempStore::new("run_seed");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let (code, out) = invoke(temp.path(), &["run", "--seed-demo"]);
    assert_eq!(code, EXIT_OK, "got: {out}");
    assert!(out.contains("warmed 1 baselines"), "got: {out}");
    assert!(out.contains("[ok] nginx"), "got: {out}");
    assert!(out.contains("[ALERT] nginx"), "got: {out}");
}

#[test]
fn run_keeps_the_audit_log_clean() {
    let temp = TempStore::new("run_audit");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let (run_code, _) = invoke(temp.path(), &["run", "--seed-demo"]);
    assert_eq!(run_code, EXIT_OK);

    let (verify_code, verify_out) = invoke(temp.path(), &["verify-store"]);
    assert_eq!(verify_code, EXIT_OK, "got: {verify_out}");
    assert!(verify_out.contains("status:  CLEAN"));
}

#[test]
fn run_without_baselines_warms_zero_and_reports_cache_misses() {
    let temp = TempStore::new("run_empty");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let (code, out) = invoke(temp.path(), &["run"]);
    assert_eq!(code, EXIT_OK, "got: {out}");
    assert!(out.contains("warmed 0 baselines"), "got: {out}");
    assert!(out.contains("[err] nginx"), "got: {out}");
}

#[test]
fn run_window_size_flag_changes_emission_count() {
    let temp = TempStore::new("run_window");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let (code, out) = invoke(
        temp.path(),
        &["run", "--seed-demo", "--window-size", "8"],
    );
    assert_eq!(code, EXIT_OK, "got: {out}");
    // 48 synthetic events / window_size=8 = 6 emissions
    let emission_lines = out
        .lines()
        .filter(|line| line.starts_with('[') && !line.starts_with("warmed"))
        .count();
    assert_eq!(emission_lines, 6, "expected 6 emissions, got: {out}");
}

#[test]
fn fit_baseline_rejects_invalid_trim_fraction() {
    let temp = TempStore::new("fit_bad_trim");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let csv = write_csv(&temp.dir, "obs.csv", &[&[0.0], &[1.0], &[2.0]]);
    let (code, _) = invoke(
        temp.path(),
        &[
            "fit-baseline",
            "p",
            "--input",
            csv.to_str().unwrap(),
            "--estimator",
            "sediment",
            "--trim-fraction",
            "1.5",
        ],
    );
    assert_eq!(code, EXIT_ERR);
}

fn write_recording(dir: &Path, name: &str, lines: &[&str]) -> PathBuf {
    let path = dir.join(name);
    let body = lines.join("\n") + "\n";
    std::fs::write(&path, body).expect("write recording fixture");
    path
}

fn synthetic_recording_lines() -> Vec<String> {
    // Mirrors daemon::synthetic_event_stream, three windows of sixteen events
    // each so the extractor closes 3 windows at the default size of 16
    let clean: [u32; 16] = [0, 0, 1, 0, 1, 3, 0, 1, 257, 1, 0, 3, 0, 1, 257, 0];
    let anomalous: [u32; 16] = [
        59, 42, 59, 56, 42, 59, 56, 42, 59, 42, 56, 59, 42, 56, 59, 42,
    ];
    let mut lines = Vec::new();
    let mut ts = 1_000_000_000u64;
    for _ in 0..2 {
        for nr in clean {
            lines.push(format!(
                "{{\"ts_ns\":{ts},\"pid\":4242,\"tgid\":4242,\"uid\":33,\
                 \"syscall_nr\":{nr},\"kind_code\":2,\"comm\":\"nginx\",\"arg0\":\"\"}}"
            ));
            ts += 1_000_000;
        }
    }
    for nr in anomalous {
        lines.push(format!(
            "{{\"ts_ns\":{ts},\"pid\":4242,\"tgid\":4242,\"uid\":33,\
             \"syscall_nr\":{nr},\"kind_code\":1,\"comm\":\"nginx\",\"arg0\":\"\"}}"
        ));
        ts += 1_000_000;
    }
    lines
}

#[test]
fn fit_from_trace_persists_a_baseline_with_replayed_observations() {
    let temp = TempStore::new("trace_fit");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let lines = synthetic_recording_lines();
    let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let recording = write_recording(&temp.dir, "trace.jsonl", &line_refs);

    let (code, out) = invoke(
        temp.path(),
        &[
            "fit-from-trace",
            "nginx",
            "--input",
            recording.to_str().unwrap(),
            "--estimator",
            "sediment",
            "--trim-fraction",
            "0.0",
        ],
    );
    assert_eq!(code, EXIT_OK, "got: {out}");
    assert!(out.contains("fit baseline for nginx"), "got: {out}");
    assert!(out.contains("3 samples"), "expected 3 windows: {out}");
    assert!(out.contains("replayed 48 events"), "got: {out}");

    let store = BaselineStore::open(temp.path()).unwrap();
    let record = store.get_baseline("nginx").unwrap().expect("baseline missing");
    assert_eq!(record.sample_count, 3);
    assert!(record.feature_dim > 0);
}

#[test]
fn fit_from_trace_errors_when_no_events_match_the_process_key() {
    let temp = TempStore::new("trace_no_match");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let lines = synthetic_recording_lines();
    let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let recording = write_recording(&temp.dir, "trace.jsonl", &line_refs);

    let (code, _) = invoke(
        temp.path(),
        &[
            "fit-from-trace",
            "missing-process",
            "--input",
            recording.to_str().unwrap(),
        ],
    );
    assert_eq!(code, EXIT_ERR);
}

#[test]
fn fit_from_trace_errors_when_window_too_large_for_event_count() {
    let temp = TempStore::new("trace_window_too_large");
    let _ = BaselineStore::open(temp.path()).unwrap();

    let lines = synthetic_recording_lines();
    let line_refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    let recording = write_recording(&temp.dir, "trace.jsonl", &line_refs);

    let (code, _) = invoke(
        temp.path(),
        &[
            "fit-from-trace",
            "nginx",
            "--input",
            recording.to_str().unwrap(),
            "--window-size",
            "1024",
        ],
    );
    assert_eq!(code, EXIT_ERR);
}
