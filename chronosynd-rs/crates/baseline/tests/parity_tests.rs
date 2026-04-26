//! Cross-implementation parity tests, loads JSON vectors emitted by
//! `chronosynd_py.parity.emit` and asserts the Rust outputs match the
//! Python reference within floating-point tolerance

use std::fs;
use std::path::PathBuf;

use chronosynd_baseline::{Baseline, NaiveBaseline, Sediment};
use ndarray::{Array1, Array2};
use serde::Deserialize;

const PARITY_VECTORS_FILE: &str = "tests/parity_vectors.json";
const ABSOLUTE_TOLERANCE: f64 = 1e-9;
const RELATIVE_TOLERANCE: f64 = 1e-9;

#[derive(Deserialize)]
struct ParityFile {
    schema_version: u32,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    estimator: Estimator,
    fit_observations: Vec<Vec<f64>>,
    score_inputs: Vec<Vec<f64>>,
    expected_scores: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Estimator {
    Naive { epsilon: f64 },
    Sediment { trim_fraction: f64, epsilon: f64 },
}

fn load_vectors() -> ParityFile {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(PARITY_VECTORS_FILE);
    let content = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "failed to read parity vectors at {}: {err}, regenerate with `uv run python -m chronosynd_py.parity.emit`",
            path.display(),
        )
    });
    serde_json::from_str(&content).expect("parity_vectors.json failed to parse")
}

fn rows_to_array2(rows: &[Vec<f64>]) -> Array2<f64> {
    let n_rows = rows.len();
    let n_cols = rows.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<f64> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    Array2::from_shape_vec((n_rows, n_cols), flat)
        .expect("vector dimensions must match the case's declared shape")
}

fn score_one(case: &Case, fit_obs: &Array2<f64>, row: ndarray::ArrayView1<'_, f64>) -> f64 {
    match case.estimator {
        Estimator::Naive { epsilon } => {
            let mut baseline = NaiveBaseline::with_epsilon(epsilon).expect("epsilon valid");
            baseline.fit(fit_obs.view()).expect("fit on parity batch");
            baseline.score(row).expect("score parity input")
        }
        Estimator::Sediment { trim_fraction, epsilon } => {
            let mut baseline =
                Sediment::with_params(trim_fraction, epsilon).expect("params valid");
            baseline.fit(fit_obs.view()).expect("fit on parity batch");
            baseline.score(row).expect("score parity input")
        }
    }
}

fn matches_within_tolerance(rust_value: f64, python_value: f64) -> bool {
    let abs_diff = (rust_value - python_value).abs();
    if abs_diff < ABSOLUTE_TOLERANCE {
        return true;
    }
    let scale = python_value.abs().max(rust_value.abs()).max(1e-300);
    abs_diff / scale < RELATIVE_TOLERANCE
}

#[test]
fn parity_with_python_reference() {
    let vectors = load_vectors();
    assert_eq!(vectors.schema_version, 1, "unsupported schema version");
    assert!(!vectors.cases.is_empty(), "parity vectors file is empty");

    for case in &vectors.cases {
        let fit_obs = rows_to_array2(&case.fit_observations);
        let score_inputs = rows_to_array2(&case.score_inputs);

        assert_eq!(
            score_inputs.nrows(),
            case.expected_scores.len(),
            "case {} declares {} score inputs but {} expected scores",
            case.name,
            score_inputs.nrows(),
            case.expected_scores.len(),
        );

        let actual: Array1<f64> = (0..score_inputs.nrows())
            .map(|i| score_one(case, &fit_obs, score_inputs.row(i)))
            .collect::<Vec<_>>()
            .into();

        for (i, (rust_score, python_score)) in actual.iter().zip(&case.expected_scores).enumerate()
        {
            assert!(
                matches_within_tolerance(*rust_score, *python_score),
                "case {} row {}: rust={} python={} abs_diff={}",
                case.name,
                i,
                rust_score,
                python_score,
                (rust_score - python_score).abs(),
            );
        }
    }
}
