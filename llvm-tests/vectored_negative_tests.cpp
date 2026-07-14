// RUN: %ppclang -O2 -fno-inline -fno-unroll-loops -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-loop-vectored -S | %FileCheck %s
//
// DVLC (Dynamic Vectored Loop Collapse) SAFETY BOUNDARY.
//
// These loops must NEVER collapse to writev/pwritev. Each keeps its
// per-iteration call (length 4096) and emits no vectored form. Because the
// string "pwritev" contains "writev", a single CHECK-NOT for "writev" rules out
// both vectored forms. (Anchoring lengths on the trailing comma keeps "4096,"
// from matching a collapsed "409600,".)
//

#include <unistd.h>
#include <sys/types.h>

#define NOINLINE __attribute__((noinline))
#define N 100

// NEG 1: stride == count is CONTIGUOUS (scalar collapse),
// so a single 409600-byte pwrite appears and NO vectored form is emitted.
// CHECK-LABEL: define {{.*}}test_stride_eq_count
// CHECK:     call{{.*}}@pwrite({{.*}}409600,
// CHECK-NOT: writev
NOINLINE void test_stride_eq_count(int fd, char *base, off_t off0) {
    for (int i = 0; i < N; i++)
        pwrite(fd, base + (long)i * 4096, 4096, off0 + (long)i * 4096);
}

// NEG 2: stride (2048) < count (4096) -> buffers OVERLAP. Deferral would read
// stale bytes. Must refuse; per-iteration pwrite survives.
// CHECK-LABEL: define {{.*}}test_overlap
// CHECK:     call{{.*}}@pwrite({{.*}}4096,
// CHECK-NOT: writev
NOINLINE void test_overlap(int fd, char *base, off_t off0) {
    for (int i = 0; i < N; i++)
        pwrite(fd, base + (long)i * 2048, 4096, off0 + (long)i * 4096);
}

// NEG 3 (CRITICAL): stride == 0 -> the SAME scratch buffer every iteration.
// Deferring would write the last iteration's contents N times. Must refuse.
// This is the corruption case; it is the most important negative in the suite.
// CHECK-LABEL: define {{.*}}test_scratch_reuse
// CHECK:     call{{.*}}@pwrite({{.*}}4096,
// CHECK-NOT: writev
NOINLINE void test_scratch_reuse(int fd, char *scratch, off_t off0) {
    for (int i = 0; i < N; i++)
        pwrite(fd, scratch, 4096, off0 + (long)i * 4096);
}

// NEG 4: return value USED. Cannot reconstruct per-iteration returns
// across a dynamic loop, so the use_empty() gate must refuse.
// CHECK-LABEL: define {{.*}}test_used_return
// CHECK:     call{{.*}}@pwrite({{.*}}4096,
// CHECK-NOT: writev
NOINLINE ssize_t test_used_return(int fd, char *base, off_t off0) {
    ssize_t total = 0;
    for (int i = 0; i < N; i++)
        total += pwrite(fd, base + (long)i * 8192, 4096, off0 + (long)i * 4096);
    return total;
}

// NEG 5: interleaved I/O on another fd. Deferring past it would reorder the
// combined transfer; the hazard scan must refuse.
// CHECK-LABEL: define {{.*}}test_interleaved_io
// CHECK:     call{{.*}}@pwrite({{.*}}4096,
// CHECK-NOT: writev
NOINLINE void test_interleaved_io(int fd, int fd2, char *base, off_t off0) {
    for (int i = 0; i < N; i++) {
        pwrite(fd, base + (long)i * 8192, 4096, off0 + (long)i * 4096);
        write(fd2, base, 1);
    }
}

// NEG 6: runtime trip count. Constant N is required (precise-range AA proof and
// a fixed iovcnt), so a symbolic n must refuse.
// CHECK-LABEL: define {{.*}}test_runtime_tripcount
// CHECK:     call{{.*}}@pwrite({{.*}}4096,
// CHECK-NOT: writev
NOINLINE void test_runtime_tripcount(int fd, char *base, off_t off0, int n) {
    for (int i = 0; i < n; i++)
        pwrite(fd, base + (long)i * 8192, 4096, off0 + (long)i * 4096);
}

// NEG 7: scattered buffer is fine, but the FILE offset stride (8192)
// != count (4096) -> non-contiguous file coverage. The pwrite offset proof must
// fail, so no pwritev is emitted.
// CHECK-LABEL: define {{.*}}test_noncontiguous_offset
// CHECK:     call{{.*}}@pwrite({{.*}}4096,
// CHECK-NOT: writev
NOINLINE void test_noncontiguous_offset(int fd, char *base, off_t off0) {
    for (int i = 0; i < N; i++)
        pwrite(fd, base + (long)i * 8192, 4096, off0 + (long)i * 8192);
}

