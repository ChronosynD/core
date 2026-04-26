# BPF programs

In-kernel programs written in restricted C, compiled to BPF bytecode, loaded into the kernel from userspace by the `chronosynd-collector` crate using Aya.

## Programs

| File | Hook | Purpose |
|---|---|---|
| `syscall_probe.bpf.c` | tracepoint:raw_syscalls:sys_enter | Records every syscall entry, classifies the kind from the syscall number |

The shared `event.h` header pins the wire format the userspace consumer reads back from the ring buffer.

## Building

The BPF C requires Linux, clang with BPF target support, libbpf headers, and a `vmlinux.h` generated from the build host's BTF:

```bash
sudo apt install -y clang llvm libbpf-dev linux-headers-$(uname -r) bpftool
bpftool btf dump file /sys/kernel/btf/vmlinux format c > bpf/vmlinux.h
clang -target bpf -O2 -g -Wall -c bpf/syscall_probe.bpf.c -o syscall_probe.bpf.o
```

`vmlinux.h` is per-host and is gitignored, every build host generates its own copy.

## Loading

The compiled object is loaded from userspace via `aya::Bpf::load`, see `chronosynd-collector/src/sources/bpf.rs` for the loader (kept Linux-only via `cfg(target_os = "linux")`). On non-Linux platforms the bpf crate exports an empty byte slice and the collector falls back to its synthetic source for testing.
