//! Build pipeline for the BPF C programs, compiles every `*.bpf.c` source
//! to ELF on Linux hosts with clang plus libbpf plus `vmlinux.h` and sets
//! the `chronosynd_bpf_compiled` cfg, no-op on every other host

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const BPF_DIR: &str = "bpf";
const SOURCES: &[&str] = &["syscall_probe.bpf.c"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for source in SOURCES {
        println!("cargo:rerun-if-changed={BPF_DIR}/{source}");
    }
    println!("cargo:rerun-if-changed={BPF_DIR}/event.h");
    println!("cargo:rerun-if-changed={BPF_DIR}/vmlinux.h");

    // Declare the custom cfg so newer rustc does not warn about it
    println!("cargo:rustc-check-cfg=cfg(chronosynd_bpf_compiled)");

    if env::var_os("CARGO_CFG_TARGET_OS").as_deref() != Some(std::ffi::OsStr::new("linux")) {
        // BPF bytecode only builds on Linux
        return;
    }

    if !clang_available() {
        println!("cargo:warning=clang not found on PATH, BPF programs will not be compiled");
        return;
    }

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let bpf_dir = crate_dir.join(BPF_DIR);
    let vmlinux = bpf_dir.join("vmlinux.h");
    if !vmlinux.exists() {
        println!(
            "cargo:warning=missing {}, run `bpftool btf dump file /sys/kernel/btf/vmlinux format c > {}` to generate it",
            vmlinux.display(),
            vmlinux.display(),
        );
        return;
    }

    if !libbpf_headers_available() {
        println!("cargo:warning=libbpf headers not found, BPF programs will not be compiled");
        println!("cargo:warning=install with `sudo apt install libbpf-dev` or equivalent");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set"));
    let mut compiled_any = false;

    for source in SOURCES {
        let source_path = bpf_dir.join(source);
        let object_name = source.trim_end_matches(".c").to_string() + ".o";
        let object_path = out_dir.join(&object_name);
        match compile_bpf_source(&source_path, &bpf_dir, &object_path) {
            Ok(()) => {
                compiled_any = true;
            }
            Err(err) => {
                println!("cargo:warning=BPF compile failed for {source}: {err}");
                return;
            }
        }
    }

    if compiled_any {
        println!("cargo:rustc-cfg=chronosynd_bpf_compiled");
    }
}

fn clang_available() -> bool {
    Command::new("clang")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

fn libbpf_headers_available() -> bool {
    Path::new("/usr/include/bpf/bpf_helpers.h").exists()
        || Path::new("/usr/local/include/bpf/bpf_helpers.h").exists()
}

fn compile_bpf_source(source: &Path, bpf_dir: &Path, output: &Path) -> Result<(), String> {
    let status = Command::new("clang")
        .args(["-target", "bpf", "-O2", "-g", "-Wall", "-c"])
        .arg("-I")
        .arg(bpf_dir)
        .arg(source)
        .arg("-o")
        .arg(output)
        .status()
        .map_err(|err| format!("invoking clang: {err}"))?;
    if !status.success() {
        return Err(format!("clang exited with {status}"));
    }
    Ok(())
}
