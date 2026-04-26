//! Pure-userspace verification of the embedded BPF object via `aya_obj`,
//! confirms structural validity (right BTF, expected program and map
//! names) without needing CAP_BPF, runs on any host with `--features bpf`

#![cfg(all(target_os = "linux", feature = "bpf"))]

use aya_obj::Object;
use chronosynd_bpf::{syscall_probe_available, SYSCALL_PROBE_OBJ};

#[test]
fn aya_obj_can_parse_the_embedded_object() {
    if !syscall_probe_available() {
        // build.rs did not produce an object on this host, skip
        return;
    }
    let object =
        Object::parse(SYSCALL_PROBE_OBJ).expect("aya-obj should parse the compiled BPF object");
    let program_names: Vec<&str> = object.programs.keys().map(String::as_str).collect();
    assert!(
        program_names.contains(&"handle_syscall"),
        "expected program 'handle_syscall' to be present, got {program_names:?}",
    );
    let map_names: Vec<&str> = object.maps.keys().map(String::as_str).collect();
    assert!(
        map_names.contains(&"events"),
        "expected map 'events' to be present, got {map_names:?}",
    );
}
