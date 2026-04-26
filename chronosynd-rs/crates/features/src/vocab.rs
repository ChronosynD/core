//! Default syscall vocabularies for the n-gram extractor, numbers are the
//! Linux x86_64 syscalls from `<asm/unistd_64.h>`, the default covers the
//! ones behavioral HIDS tools typically watch (process, file, network)

/// Reasonable default syscall vocabulary for Linux x86_64, covers the
/// syscalls a process-behavior baseline most commonly relies on, anything
/// outside this set falls into the extractor's catch-all "other" bucket
pub fn default_syscall_vocab() -> Vec<u32> {
    vec![
        0,   // read
        1,   // write
        2,   // open
        3,   // close
        4,   // stat
        5,   // fstat
        9,   // mmap
        10,  // mprotect
        11,  // munmap
        21,  // access
        41,  // socket
        42,  // connect
        43,  // accept
        44,  // sendto
        45,  // recvfrom
        47,  // recvmsg
        56,  // clone
        57,  // fork
        59,  // execve
        60,  // exit
        62,  // kill
        87,  // unlink
        90,  // chmod
        92,  // chown
        232, // epoll_wait
        257, // openat
    ]
}
