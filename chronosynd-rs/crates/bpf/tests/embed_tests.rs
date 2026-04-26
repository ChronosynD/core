//! Verifies the build pipeline embedded a real BPF object when expected,
//! checks ELF magic and the EM_BPF machine code on Linux with the
//! toolchain present, passes trivially on every other host

use chronosynd_bpf::{syscall_probe_available, SYSCALL_PROBE_OBJ};

#[test]
fn empty_object_means_pipeline_was_skipped() {
    if !syscall_probe_available() {
        assert!(SYSCALL_PROBE_OBJ.is_empty());
    }
}

#[test]
fn embedded_object_starts_with_elf_magic_when_compiled() {
    if syscall_probe_available() {
        // ELF magic: 0x7F 'E' 'L' 'F'
        assert_eq!(&SYSCALL_PROBE_OBJ[..4], &[0x7F, b'E', b'L', b'F']);
    }
}

#[test]
fn embedded_object_targets_ebpf_machine_when_compiled() {
    if syscall_probe_available() {
        // e_machine sits at offset 0x12, value 247 is EM_BPF in the ELF spec
        let machine_low = SYSCALL_PROBE_OBJ[0x12];
        let machine_high = SYSCALL_PROBE_OBJ[0x13];
        let machine = u16::from_le_bytes([machine_low, machine_high]);
        assert_eq!(machine, 247, "expected EM_BPF (247) but got {machine}");
    }
}
