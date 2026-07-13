// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s --check-prefix=HOIST
// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-assume-regular-files=false -S | %FileCheck %s --check-prefix=NOHOIST

#include <unistd.h>

extern "C" {

// A constant-trip-count loop of fixed-size, contiguous writes is hoistable: the
// pass collapses 100 iterations of a 16-byte write into one 100*16 = 1600-byte
// write in the loop exit block. Whether it does so must now depend on
// -io-opt-assume-regular-files, exactly like the batching pass.

// HOIST-LABEL: define {{.*}}@hoistable_writes
// NOHOIST-LABEL: define {{.*}}@hoistable_writes
void hoistable_writes(int fd, char* buf) {
    // Default (assume regular files): the loop body write is hoisted and merged
    // into a single 1600-byte write; no per-iteration 16-byte write survives.
    // HOIST: call i64 @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}1600)
    // HOIST-NOT: i64 {{.*}}16)

    // With hoisting disabled for non-regular fds, merging could change PIPE_BUF
    // atomicity / datagram message boundaries, so the transform is suppressed:
    // the original per-iteration 16-byte write must survive and no 1600-byte
    // hoisted write may appear.
    // NOHOIST: call i64 @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}16)
    // NOHOIST-NOT: i64 {{.*}}1600)
    for (int i = 0; i < 100; i++) {
        write(fd, buf + (i * 16), 16);
    }
}

}

