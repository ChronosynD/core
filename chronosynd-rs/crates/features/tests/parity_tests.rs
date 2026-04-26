//! Cross-implementation parity tests for the syscall n-gram extractor,
//! loads JSON vectors emitted by `chronosynd_py.parity.emit_features` and
//! asserts the Rust extractor produces byte-identical feature vectors

use std::fs;
use std::path::PathBuf;

use chronosynd_features::SyscallNgramExtractor;
use serde::Deserialize;

const PARITY_VECTORS_FILE: &str = "tests/parity_vectors.json";
const ABSOLUTE_TOLERANCE: f64 = 1e-12;

#[derive(Deserialize)]
struct ParityFile {
    schema_version: u32,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    window_size: usize,
    vocab: Vec<u32>,
    events: Vec<EventEntry>,
    emissions: Vec<EmissionEntry>,
}

#[derive(Deserialize)]
struct EventEntry {
    process_key: String,
    syscall_nr: u32,
}

#[derive(Deserialize)]
struct EmissionEntry {
    event_index: usize,
    process_key: String,
    feature_vector: Vec<f64>,
}

fn load_vectors() -> ParityFile {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(PARITY_VECTORS_FILE);
    let content = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "Could not read parity vectors at {}, regenerate with \
             `uv run python -m chronosynd_py.parity.emit_features`: {err}",
            path.display()
        )
    });
    serde_json::from_str(&content).expect("parity vectors must parse as JSON")
}

#[test]
fn parity_with_python_reference() {
    let file = load_vectors();
    assert_eq!(file.schema_version, 1, "schema version must match the emitter");

    let mut total_emissions = 0usize;
    for case in &file.cases {
        let mut extractor = SyscallNgramExtractor::new(case.vocab.clone(), case.window_size)
            .unwrap_or_else(|err| panic!("case {}: build extractor: {err}", case.name));

        let mut actual_emissions: Vec<(usize, String, Vec<f64>)> = Vec::new();
        for (index, event) in case.events.iter().enumerate() {
            if let Some(emitted) = extractor.accumulate(&event.process_key, event.syscall_nr) {
                actual_emissions.push((index, emitted.process_key, emitted.feature_vector));
            }
        }

        assert_eq!(
            actual_emissions.len(),
            case.emissions.len(),
            "case {}: emission count diverges, Rust={} Python={}",
            case.name,
            actual_emissions.len(),
            case.emissions.len(),
        );

        for (rust_emission, py_emission) in
            actual_emissions.iter().zip(case.emissions.iter())
        {
            let (rust_index, rust_key, rust_features) = rust_emission;
            assert_eq!(
                *rust_index, py_emission.event_index,
                "case {}: emission event index diverges, Rust={} Python={}",
                case.name, rust_index, py_emission.event_index,
            );
            assert_eq!(
                rust_key, &py_emission.process_key,
                "case {}: emission process_key diverges at event {}, Rust={:?} Python={:?}",
                case.name, rust_index, rust_key, py_emission.process_key,
            );
            assert_eq!(
                rust_features.len(),
                py_emission.feature_vector.len(),
                "case {}: feature_vector length diverges at event {}",
                case.name,
                rust_index,
            );
            for (axis, (rust_value, py_value)) in rust_features
                .iter()
                .zip(py_emission.feature_vector.iter())
                .enumerate()
            {
                let diff = (rust_value - py_value).abs();
                assert!(
                    diff <= ABSOLUTE_TOLERANCE,
                    "case {}: feature {} diverges at event {}, Rust={} Python={} diff={}",
                    case.name,
                    axis,
                    rust_index,
                    rust_value,
                    py_value,
                    diff,
                );
            }
            total_emissions += 1;
        }
    }

    assert!(total_emissions > 0, "expected at least one emission across cases");
}
