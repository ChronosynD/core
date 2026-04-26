//! Tests for the syscall n-gram extractor

use chronosynd_features::{
    default_syscall_vocab, EmittedFeatures, FeatureError, SyscallNgramExtractor,
};

#[test]
fn feature_dim_is_vocab_plus_other_bucket() {
    let vocab = vec![0, 1, 2];
    let extractor = SyscallNgramExtractor::new(vocab, 4).unwrap();
    assert_eq!(extractor.feature_dim(), 4);
    assert_eq!(extractor.other_bucket_index(), 3);
}

#[test]
fn accumulating_below_window_size_returns_none() {
    let mut extractor = SyscallNgramExtractor::new(vec![0, 1], 4).unwrap();
    assert!(extractor.accumulate("nginx", 0).is_none());
    assert!(extractor.accumulate("nginx", 1).is_none());
    assert!(extractor.accumulate("nginx", 0).is_none());
}

#[test]
fn closing_window_emits_normalized_histogram() {
    let mut extractor = SyscallNgramExtractor::new(vec![0, 1, 2], 4).unwrap();
    extractor.accumulate("nginx", 0).unwrap_or(default_emit());
    extractor.accumulate("nginx", 0).unwrap_or(default_emit());
    extractor.accumulate("nginx", 1).unwrap_or(default_emit());
    let emitted = extractor.accumulate("nginx", 2).expect("window should close");
    assert_eq!(emitted.process_key, "nginx");
    assert_eq!(emitted.feature_vector, vec![0.5, 0.25, 0.25, 0.0]);

    let sum: f64 = emitted.feature_vector.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

#[test]
fn out_of_vocabulary_syscalls_go_to_the_other_bucket() {
    let mut extractor = SyscallNgramExtractor::new(vec![0, 1], 4).unwrap();
    let _ = extractor.accumulate("p", 0);
    let _ = extractor.accumulate("p", 999);
    let _ = extractor.accumulate("p", 9999);
    let emitted = extractor.accumulate("p", 0).expect("window should close");
    assert_eq!(emitted.feature_vector, vec![0.5, 0.0, 0.5]);
}

#[test]
fn each_process_has_its_own_independent_window() {
    let mut extractor = SyscallNgramExtractor::new(vec![0, 1, 2], 2).unwrap();
    assert!(extractor.accumulate("a", 0).is_none());
    assert!(extractor.accumulate("b", 1).is_none());
    let emit_a = extractor.accumulate("a", 1).expect("a should close");
    let emit_b = extractor.accumulate("b", 2).expect("b should close");
    assert_eq!(emit_a.process_key, "a");
    assert_eq!(emit_a.feature_vector, vec![0.5, 0.5, 0.0, 0.0]);
    assert_eq!(emit_b.process_key, "b");
    assert_eq!(emit_b.feature_vector, vec![0.0, 0.5, 0.5, 0.0]);
}

#[test]
fn second_window_starts_clean_after_emission() {
    let mut extractor = SyscallNgramExtractor::new(vec![0, 1], 2).unwrap();
    let _ = extractor.accumulate("p", 0);
    let first = extractor.accumulate("p", 0).expect("first close");
    assert_eq!(first.feature_vector, vec![1.0, 0.0, 0.0]);

    let _ = extractor.accumulate("p", 1);
    let second = extractor.accumulate("p", 1).expect("second close");
    assert_eq!(second.feature_vector, vec![0.0, 1.0, 0.0]);
}

#[test]
fn reset_process_drops_any_pending_window() {
    let mut extractor = SyscallNgramExtractor::new(vec![0, 1], 3).unwrap();
    let _ = extractor.accumulate("p", 0);
    let _ = extractor.accumulate("p", 0);
    extractor.reset_process("p");
    assert!(extractor.accumulate("p", 0).is_none());
    let _ = extractor.accumulate("p", 0);
    let emitted = extractor.accumulate("p", 0).expect("window should close after reset");
    assert_eq!(emitted.feature_vector, vec![1.0, 0.0, 0.0]);
}

#[test]
fn tracked_process_count_grows_per_seen_process() {
    let mut extractor = SyscallNgramExtractor::new(vec![0], 4).unwrap();
    assert_eq!(extractor.tracked_process_count(), 0);
    let _ = extractor.accumulate("a", 0);
    let _ = extractor.accumulate("b", 0);
    let _ = extractor.accumulate("a", 0);
    assert_eq!(extractor.tracked_process_count(), 2);
}

#[test]
fn zero_window_size_is_rejected() {
    let err = SyscallNgramExtractor::new(vec![0, 1], 0).unwrap_err();
    assert!(matches!(err, FeatureError::InvalidParameter(_)));
}

#[test]
fn duplicate_vocab_entry_is_rejected() {
    let err = SyscallNgramExtractor::new(vec![0, 1, 0], 4).unwrap_err();
    assert!(matches!(err, FeatureError::DuplicateVocabEntry(0)));
}

#[test]
fn default_vocab_returns_a_meaningful_set() {
    let vocab = default_syscall_vocab();
    assert!(vocab.contains(&0)); // read
    assert!(vocab.contains(&1)); // write
    assert!(vocab.contains(&59)); // execve
    assert!(vocab.contains(&257)); // openat
    let mut sorted = vocab.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), vocab.len(), "default vocab must be unique");
}

#[test]
fn extractor_built_from_default_vocab_has_consistent_dim() {
    let vocab = default_syscall_vocab();
    let expected = vocab.len() + 1;
    let extractor = SyscallNgramExtractor::new(vocab, 16).unwrap();
    assert_eq!(extractor.feature_dim(), expected);
}

fn default_emit() -> EmittedFeatures {
    EmittedFeatures {
        process_key: String::new(),
        feature_vector: Vec::new(),
    }
}
