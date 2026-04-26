//! Diagnostic example, prints whether a real BPF object is embedded,
//! Linux hosts with clang plus libbpf show non-zero size and ELF magic,
//! every other host shows zero and a hint to install the toolchain

use chronosynd_bpf::{syscall_probe_available, SYSCALL_PROBE_OBJ};

fn main() {
    let bytes = SYSCALL_PROBE_OBJ;
    println!("syscall_probe_available: {}", syscall_probe_available());
    println!("SYSCALL_PROBE_OBJ.len(): {}", bytes.len());
    if bytes.len() >= 4 {
        let magic = &bytes[..4];
        println!("first 4 bytes (hex): {:02X} {:02X} {:02X} {:02X}", magic[0], magic[1], magic[2], magic[3]);
        if magic == [0x7F, b'E', b'L', b'F'] {
            println!("ELF magic confirmed, this is a compiled BPF object");
        }
    } else {
        println!("constant is empty, build.rs did not run a BPF compile on this host");
    }
}
