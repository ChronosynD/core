// Shared event layout for the BPF probes and the userspace consumer
//
// The struct is laid out C-style and zero-padded on the BPF side so the
// userspace can rely on a fixed 96-byte record per event in the ring
// buffer, the field order is held stable as part of the ABI between the
// kernel and userspace halves of the collector
//
// This header expects __u32 and __u64 to be in scope already, the .bpf.c
// includer pulls vmlinux.h before this file which supplies those types,
// pulling <linux/types.h> here would drag in <asm/types.h> and the
// cross-arch userspace headers we deliberately do not depend on

#ifndef CHRONOSYND_EVENT_H
#define CHRONOSYND_EVENT_H

#define CHRONOSYND_COMM_MAX 16
#define CHRONOSYND_ARG_MAX 64

enum chronosynd_event_kind {
    CHRONOSYND_EVENT_EXEC = 1,
    CHRONOSYND_EVENT_FILE_OPEN = 2,
    CHRONOSYND_EVENT_NET_CONNECT = 3,
    CHRONOSYND_EVENT_PROCESS_EXIT = 4,
    CHRONOSYND_EVENT_OTHER_SYSCALL = 5,
};

struct chronosynd_event {
    __u64 ts_ns;
    __u32 pid;
    __u32 tgid;
    __u32 uid;
    __u32 syscall_nr;
    __u32 kind;
    __u32 _padding;
    char comm[CHRONOSYND_COMM_MAX];
    char arg0[CHRONOSYND_ARG_MAX];
};

#endif
