//! Unit tests for the synthetic event source

use chronosynd_collector::sources::synthetic::SyntheticSource;
use chronosynd_collector::{Event, EventKind, EventSource, EventSourceError};

fn sample_events() -> Vec<Event> {
    vec![
        Event {
            ts_ns: 100,
            pid: 1,
            tgid: 1,
            uid: 0,
            syscall_nr: 59,
            kind: EventKind::Exec,
            comm: "init".into(),
            arg0: "/sbin/init".into(),
        },
        Event {
            ts_ns: 200,
            pid: 2,
            tgid: 2,
            uid: 1000,
            syscall_nr: 41,
            kind: EventKind::NetConnect,
            comm: "curl".into(),
            arg0: "1.2.3.4:443".into(),
        },
    ]
}

#[test]
fn yields_events_in_order() {
    let mut source = SyntheticSource::new(sample_events());
    let first = source.next_event().expect("first event");
    let second = source.next_event().expect("second event");
    assert_eq!(first.pid, 1);
    assert_eq!(second.pid, 2);
}

#[test]
fn signals_closed_when_exhausted() {
    let mut source = SyntheticSource::new(sample_events());
    let _ = source.next_event().unwrap();
    let _ = source.next_event().unwrap();
    let err = source.next_event().unwrap_err();
    assert!(matches!(err, EventSourceError::Closed));
}

#[test]
fn empty_source_signals_closed_immediately() {
    let mut source = SyntheticSource::new(Vec::new());
    let err = source.next_event().unwrap_err();
    assert!(matches!(err, EventSourceError::Closed));
}

#[test]
fn remaining_decrements_with_each_event() {
    let mut source = SyntheticSource::new(sample_events());
    assert_eq!(source.remaining(), 2);
    source.next_event().unwrap();
    assert_eq!(source.remaining(), 1);
    source.next_event().unwrap();
    assert_eq!(source.remaining(), 0);
}

#[test]
fn event_kind_from_code_round_trips_known_values() {
    use chronosynd_bpf::{
        RAW_EVENT_KIND_EXEC, RAW_EVENT_KIND_FILE_OPEN, RAW_EVENT_KIND_NET_CONNECT,
        RAW_EVENT_KIND_PROCESS_EXIT,
    };
    assert_eq!(EventKind::from_code(RAW_EVENT_KIND_EXEC), EventKind::Exec);
    assert_eq!(EventKind::from_code(RAW_EVENT_KIND_FILE_OPEN), EventKind::FileOpen);
    assert_eq!(
        EventKind::from_code(RAW_EVENT_KIND_NET_CONNECT),
        EventKind::NetConnect,
    );
    assert_eq!(
        EventKind::from_code(RAW_EVENT_KIND_PROCESS_EXIT),
        EventKind::ProcessExit,
    );
}

#[test]
fn event_kind_from_code_passes_through_unknown_codes() {
    assert_eq!(EventKind::from_code(9999), EventKind::Unknown(9999));
}
