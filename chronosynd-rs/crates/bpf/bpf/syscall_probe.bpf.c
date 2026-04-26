// Generic syscall-entry probe. Attaches to `raw_syscalls/sys_enter` so it
// fires once per syscall regardless of which syscall it is, classifies
// the event kind based on the syscall number, and emits the syscall id
// into `syscall_nr` for the userspace n-gram histogram to bucket
//
// Build target is BPF bytecode produced by `clang -target bpf -O2 -g -c`,
// the resulting object file is loaded from userspace via Aya, see
// `chronosynd-collector` for the loader

// vmlinux.h carries the kernel BTF declarations the BPF programs need,
// generate it once on the build host with:
//   bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#include "event.h"

char LICENSE[] SEC("license") = "GPL";

// x86_64 syscall numbers we recognize for richer kind classification, the
// list mirrors the userspace vocab in `chronosynd-features/src/vocab.rs`
#define SYS_OPEN          2
#define SYS_OPENAT        257
#define SYS_CONNECT       42
#define SYS_EXECVE        59
#define SYS_EXIT          60
#define SYS_EXIT_GROUP    231

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 20);
} events SEC(".maps");

static __always_inline __u32 classify_kind(__u32 syscall_nr)
{
    if (syscall_nr == SYS_EXECVE)
        return CHRONOSYND_EVENT_EXEC;
    if (syscall_nr == SYS_OPEN || syscall_nr == SYS_OPENAT)
        return CHRONOSYND_EVENT_FILE_OPEN;
    if (syscall_nr == SYS_CONNECT)
        return CHRONOSYND_EVENT_NET_CONNECT;
    if (syscall_nr == SYS_EXIT || syscall_nr == SYS_EXIT_GROUP)
        return CHRONOSYND_EVENT_PROCESS_EXIT;
    return CHRONOSYND_EVENT_OTHER_SYSCALL;
}

SEC("tracepoint/raw_syscalls/sys_enter")
int handle_syscall(struct trace_event_raw_sys_enter *ctx)
{
    struct chronosynd_event *event;
    __u32 syscall_nr = (__u32)ctx->id;

    event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    __u64 pid_tgid = bpf_get_current_pid_tgid();
    event->ts_ns = bpf_ktime_get_ns();
    event->pid = (__u32)pid_tgid;
    event->tgid = (__u32)(pid_tgid >> 32);
    event->uid = (__u32)bpf_get_current_uid_gid();
    event->syscall_nr = syscall_nr;
    event->kind = classify_kind(syscall_nr);
    event->_padding = 0;

    bpf_get_current_comm(&event->comm, sizeof(event->comm));

    // Zero the full arg0 buffer before any conditional fill. The ring
    // buffer does not zero memory on reserve, so without this the bytes
    // after the first NUL could carry stale data from a prior reservation
    // (defense in depth, the userspace string parser already stops at NUL)
    #pragma unroll
    for (__u32 i = 0; i < CHRONOSYND_ARG_MAX; i++) {
        event->arg0[i] = 0;
    }

    // arg0 carries the filename for execve. The first arg of every other
    // syscall in our vocab means something different, so a generic read
    // would just produce noise. Leave arg0 empty for non-exec syscalls
    if (syscall_nr == SYS_EXECVE) {
        const char *filename = (const char *)ctx->args[0];
        bpf_probe_read_user_str(&event->arg0, sizeof(event->arg0), filename);
    }

    bpf_ringbuf_submit(event, 0);
    return 0;
}
