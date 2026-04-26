"""Default syscall vocabularies for the n-gram extractor. The numbers are
the Linux x86_64 syscalls from `<asm/unistd_64.h>`, kept in lockstep with
`chronosynd-rs/crates/features/src/vocab.rs` and parity-checked on push"""

from __future__ import annotations


def default_syscall_vocab() -> list[int]:
    """Reasonable default syscall vocabulary for Linux x86_64. Covers the
    syscalls a process-behavior baseline most commonly relies on. Anything
    outside this set falls into the extractor's catch-all "other" bucket"""
    return [
        0,    # read
        1,    # write
        2,    # open
        3,    # close
        4,    # stat
        5,    # fstat
        9,    # mmap
        10,   # mprotect
        11,   # munmap
        21,   # access
        41,   # socket
        42,   # connect
        43,   # accept
        44,   # sendto
        45,   # recvfrom
        47,   # recvmsg
        56,   # clone
        57,   # fork
        59,   # execve
        60,   # exit
        62,   # kill
        87,   # unlink
        90,   # chmod
        92,   # chown
        232,  # epoll_wait
        257,  # openat
    ]
