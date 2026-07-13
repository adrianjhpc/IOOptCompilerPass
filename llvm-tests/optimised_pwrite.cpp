// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>
#include <sys/types.h>

extern "C" {

// Four scattered buffers would normally be a pwritev candidate, but each
// pwrite's return value is checked with `if (rN < 0) return;` before the next
// pwrite runs. Because those results are consumed before the merge point, the
// return-availability gate refuses the batch: merging could perform later
// writes the original program would have skipped after an error. The calls are
// left intact and no iovec array is emitted.

// CHECK-LABEL: define {{.*}}optimised_pwrite
void optimised_pwrite(int fd, const char* b1, const char* b2, const char* b3, const char* b4, size_t len, off_t offset) {
    // No batching: all four original pwrite calls must survive.
    // CHECK: call i64 @pwrite(
    // CHECK: call i64 @pwrite(
    // CHECK: call i64 @pwrite(
    // CHECK: call i64 @pwrite(

    // The pwritev conversion and its iovec array must NOT be generated.
    // CHECK-NOT: iovec.array.N
    // CHECK-NOT: call i64 @pwritev

    ssize_t r1 = pwrite(fd, b1, len, offset);
    if (r1 < 0) return;

    ssize_t r2 = pwrite(fd, b2, len, offset + len);
    if (r2 < 0) return;

    ssize_t r3 = pwrite(fd, b3, len, offset + (len * 2));
    if (r3 < 0) return;

    ssize_t r4 = pwrite(fd, b4, len, offset + (len * 3));
    if (r4 < 0) return;
}

}

