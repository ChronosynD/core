//! End-to-end tests for record mode and replay-and-fit, both round-trip
//! through the JSONL wire format and the SyscallNgramExtractor

use std::path::{Path, PathBuf};

use chronosynd_collector::{
    extract_observations_from_recording, read_recording, run_daemon, EXIT_OK,
};

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(label: &str) -> Self {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "chronosynd_record_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    fn child(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn invoke(store: &Path, extra_args: &[&str]) -> i32 {
    let mut argv: Vec<String> = vec![
        "chronosynd-collector".into(),
        "--store".into(),
        store.to_string_lossy().into_owned(),
    ];
    argv.extend(extra_args.iter().map(|s| s.to_string()));
    let mut buffer = Vec::new();
    run_daemon(argv, &mut buffer)
}

#[test]
fn record_writes_one_jsonl_line_per_synthetic_event() {
    let temp = TempDir::new("write");
    let store = temp.child("chronosynd.db");
    let recording = temp.child("trace.jsonl");

    let code = invoke(
        &store,
        &[
            "--seed-demo",
            "--record",
            recording.to_str().unwrap(),
        ],
    );
    assert_eq!(code, EXIT_OK);

    let contents = std::fs::read_to_string(&recording).expect("recording exists");
    let line_count = contents.lines().filter(|line| !line.trim().is_empty()).count();
    assert_eq!(
        line_count, 48,
        "synthetic stream emits 48 events (3 windows of 16), got {line_count}"
    );
}

#[test]
fn recorded_events_round_trip_through_read_recording() {
    let temp = TempDir::new("roundtrip");
    let store = temp.child("chronosynd.db");
    let recording = temp.child("trace.jsonl");

    let code = invoke(
        &store,
        &["--seed-demo", "--record", recording.to_str().unwrap()],
    );
    assert_eq!(code, EXIT_OK);

    let events = read_recording(&recording).expect("recording reads cleanly");
    assert_eq!(events.len(), 48);
    assert!(events.iter().all(|event| event.comm == "nginx"));
}

#[test]
fn replay_emits_three_observation_rows_at_default_window() {
    let temp = TempDir::new("replay");
    let store = temp.child("chronosynd.db");
    let recording = temp.child("trace.jsonl");

    let code = invoke(
        &store,
        &["--seed-demo", "--record", recording.to_str().unwrap()],
    );
    assert_eq!(code, EXIT_OK);

    let replay = extract_observations_from_recording(&recording, "nginx", 16)
        .expect("replay succeeds");
    assert_eq!(replay.events_for_process, 48);
    assert_eq!(replay.rows.len(), 3, "48 events at window=16 yield 3 rows");
    let feature_dim = replay.rows[0].len();
    assert!(replay.rows.iter().all(|row| row.len() == feature_dim));
}

#[test]
fn replay_filters_out_non_matching_process_keys() {
    let temp = TempDir::new("filter");
    let store = temp.child("chronosynd.db");
    let recording = temp.child("trace.jsonl");

    let code = invoke(
        &store,
        &["--seed-demo", "--record", recording.to_str().unwrap()],
    );
    assert_eq!(code, EXIT_OK);

    let replay = extract_observations_from_recording(&recording, "not-this-process", 16)
        .expect("replay handles empty match cleanly");
    assert_eq!(replay.events_for_process, 0);
    assert_eq!(replay.rows.len(), 0);
    assert_eq!(replay.total_events_read, 48);
}
