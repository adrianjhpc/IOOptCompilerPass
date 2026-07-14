// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s --check-prefix=SEQ
// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-prefetch-sequential=false -S | %FileCheck %s --check-prefix=NOSEQ

#include <unistd.h>

extern "C" {

// A read() loop into a FIXED buffer: the file offset advances implicitly, so
// it's sequential, but the buffer is not an addrec -> NOT hoistable and NOT a
// WILLNEED target (implicit offset). This is exactly what SEQUENTIAL covers.

// SEQ-LABEL: define {{.*}}@seq_read_loop
// NOSEQ-LABEL: define {{.*}}@seq_read_loop
void seq_read_loop(int fd, char* buf) {
    // SEQ: call i32 @posix_fadvise(i32 {{.*}}, i64 0, i64 0, i32 2)
    // SEQ: call i64 @read(
    // NOSEQ-NOT: posix_fadvise
    for (int i = 0; i < 1000; i++) {
        read(fd, buf, 4096);
    }
}

}

