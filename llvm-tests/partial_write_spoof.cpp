// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// The two writes look contiguous in the fd, but the return value of the first
// write is consumed by the `if (w1 < 10)` branch *before* the second write
// executes. Merging into a single 20-byte write would push buf2's bytes to the
// fd even when the original program would have bailed out after a short write1.
// The return-availability gate therefore (correctly) refuses to merge: both
// writes are left intact and no shadow buffer is created.

// CHECK-LABEL: define dso_local noundef zeroext i1 @_Z21test_short_write_trapiPcS_
bool test_short_write_trap(int fd, char* buf1, char* buf2) {
    // No batching: the original per-call writes must survive verbatim.
    // CHECK: call i64 @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}10)
    // CHECK: call i64 @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}10)

    // And none of the merge machinery should appear.
    // CHECK-NOT: alloca [20 x i8]
    // CHECK-NOT: shadow.buf
    // CHECK-NOT: spoofed.posix.ret
    // CHECK-NOT: io.posix.ret

    ssize_t w1 = write(fd, buf1, 10);
    if (w1 < 10) return false;

    ssize_t w2 = write(fd, buf2, 10);
    if (w2 < 10) return false;

    return true;
}

