//! Integration tests for the collector daemon

use std::path::{Path, PathBuf};

use chronosynd_collector::{run_daemon, EXIT_ERR, EXIT_OK};

struct TempStore {
    dir: PathBuf,
    path: PathBuf,
}

impl TempStore {
    fn new(label: &str) -> Self {
        let mut dir = std::env::temp_dir();
        let unique = format!(
            "chronosynd_daemon_{label}_{}_{}",
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

fn invoke(store: &Path, extra_args: &[&str]) -> (i32, String) {
    let mut argv: Vec<String> = vec!["chronosynd-collector".into()];
    argv.push("--store".into());
    argv.push(store.to_string_lossy().into_owned());
    argv.extend(extra_args.iter().map(|s| s.to_string()));
    let mut buffer = Vec::new();
    let code = run_daemon(argv, &mut buffer);
    (code, String::from_utf8(buffer).expect("daemon output is utf-8"))
}

#[test]
fn empty_store_warms_zero_baselines_and_reports_cache_misses() {
    let temp = TempStore::new("empty");
    let (code, out) = invoke(temp.path(), &[]);
    assert_eq!(code, EXIT_OK);
    assert!(out.contains("warmed 0 baselines"), "output was: {out}");
    assert!(out.contains("[err]"), "expected cache-miss errors: {out}");
}

#[test]
fn seed_demo_produces_one_baseline_and_three_emissions() {
    let temp = TempStore::new("seeded");
    let (code, out) = invoke(temp.path(), &["--seed-demo"]);
    assert_eq!(code, EXIT_OK);
    assert!(
        out.contains("warmed 1 baselines"),
        "expected one baseline warmed, got: {out}"
    );
    let nginx_lines: Vec<&str> = out
        .lines()
        .filter(|line| line.contains("nginx"))
        .collect();
    assert_eq!(
        nginx_lines.len(),
        3,
        "expected three emission lines for nginx, got: {nginx_lines:?}"
    );
}

#[test]
fn clean_windows_score_below_threshold_anomalous_window_alerts() {
    let temp = TempStore::new("alert");
    let (code, out) =
        invoke(temp.path(), &["--seed-demo", "--default-threshold", "100"]);
    assert_eq!(code, EXIT_OK);
    let nginx_lines: Vec<&str> = out
        .lines()
        .filter(|line| line.contains("nginx"))
        .collect();
    assert_eq!(nginx_lines.len(), 3);

    let ok_count = nginx_lines.iter().filter(|line| line.starts_with("[ok]")).count();
    let alert_count = nginx_lines
        .iter()
        .filter(|line| line.starts_with("[ALERT]"))
        .count();
    assert_eq!(ok_count, 2, "expected two clean lines: {nginx_lines:?}");
    assert_eq!(alert_count, 1, "expected one alert line: {nginx_lines:?}");
}

#[test]
fn second_run_after_seed_does_not_duplicate_baselines() {
    let temp = TempStore::new("idempotent");
    let _ = invoke(temp.path(), &["--seed-demo"]);
    let (code, out) = invoke(temp.path(), &["--seed-demo"]);
    assert_eq!(code, EXIT_OK);
    assert!(
        out.contains("warmed 1 baselines"),
        "expected single baseline after re-seed, got: {out}"
    );
}

#[test]
fn unknown_argument_returns_error_code() {
    let temp = TempStore::new("badarg");
    let (code, _) = invoke(temp.path(), &["--definitely-not-a-flag"]);
    assert_eq!(code, EXIT_ERR);
}

#[test]
fn window_size_flag_is_respected() {
    // window_size=8 with 48 synthetic events should give exactly 6 emissions
    let temp = TempStore::new("window_size");
    let (code, out) = invoke(
        temp.path(),
        &["--seed-demo", "--window-size", "8", "--default-threshold", "100"],
    );
    assert_eq!(code, EXIT_OK);
    let total: usize = out
        .lines()
        .filter(|line| line.contains("nginx"))
        .count();
    assert_eq!(total, 6, "expected six emissions at window=8, got: {out}");
}
