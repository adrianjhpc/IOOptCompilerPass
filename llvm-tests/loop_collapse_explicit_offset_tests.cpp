// RUN: %ppclang -O2 -fno-inline -fno-unroll-loops -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s
//
// Tier-1 explicit-offset loop collapse (IOLoopHoistingPass).
//
// A loop that issues one pread/pwrite per iteration at a contiguous, unit-stride
// file offset (offset_i = start + i*count) over a contiguous, unit-stride buffer
// is collapsed into a SINGLE whole-range transfer:
//   * writes  -> lowered to the loop EXIT block (buffers must be filled first)
//   * reads   -> hoisted to the loop PREHEADER  (data must exist before the loop)
//
// -fno-unroll-loops keeps the loop as a loop so IOLoopHoistingPass (not the
// straight-line IOBatchingPass) is what we exercise. Positive cases use a
// COMPILE-TIME trip count because Tier-1 requires a constant backedge count.
//
// Detection metric: the per-iteration length (4096) is replaced by N*4096.
// For N = 100 the collapsed length is 409600. Note the merged call is built by
// the pass (no 'noundef' attrs); the original per-iteration calls carry them,
// hence the flexible {{.*}} patterns below. We anchor on the trailing comma so
// "4096," never matches "409600,".

#include <unistd.h>
#include <sys/types.h>

#define NOINLINE __attribute__((noinline))
#define N 100

// ---------------------------------------------------------------------------
// POSITIVE 1: contiguous pwrite loop -> one pwrite of N*count in the EXIT block.
// The 'ret void' immediately after the merged call proves EXIT-block placement
// (a deferred write is followed only by the terminator).
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_pwrite_collapse
// CHECK:      call i64 @pwrite({{.*}}409600,
// CHECK-NEXT: ret void
// CHECK-NOT:  call i64 @pwrite
NOINLINE void test_pwrite_collapse(int fd, char *base, off_t off0) {
    for (int i = 0; i < N; i++)
        pwrite(fd, base + (long)i * 4096, 4096, off0 + (long)i * 4096);
}

// ---------------------------------------------------------------------------
// POSITIVE 2: contiguous pread loop -> one pread of N*count in the PREHEADER.
// Exactly one pread call remains.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_pread_collapse
// CHECK:     call i64 @pread({{.*}}409600,
// CHECK-NOT: call i64 @pread
NOINLINE void test_pread_collapse(int fd, char *base, off_t off0) {
    for (int i = 0; i < N; i++)
        pread(fd, base + (long)i * 4096, 4096, off0 + (long)i * 4096);
}

// ---------------------------------------------------------------------------
// NEGATIVE 1: file offset stride (8192) != count (4096). Not contiguous, so the
// offset-addrec proof must FAIL: the per-iteration 4096 call survives, and no
// collapsed 409600 call appears.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_noncontiguous_offset
// CHECK:     call i64 @pwrite({{.*}}4096,
// CHECK-NOT: 409600
NOINLINE void test_noncontiguous_offset(int fd, char *base, off_t off0) {
    for (int i = 0; i < N; i++)
        pwrite(fd, base + (long)i * 4096, 4096, off0 + (long)i * 8192);
}

// ---------------------------------------------------------------------------
// NEGATIVE 2: the pwrite return value is USED (running total). Tier-1 cannot
// reconstruct per-iteration returns, so the use_empty() gate must refuse.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_used_return
// CHECK:     call i64 @pwrite({{.*}}4096,
// CHECK-NOT: 409600
NOINLINE ssize_t test_used_return(int fd, char *base, off_t off0) {
    ssize_t total = 0;
    for (int i = 0; i < N; i++)
        total += pwrite(fd, base + (long)i * 4096, 4096, off0 + (long)i * 4096);
    return total;
}

// ---------------------------------------------------------------------------
// NEGATIVE 3: a second, interleaved I/O call in the loop body. Deferring the
// pwrite would reorder it past that call, so the hazard scan must refuse.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_interleaved_io
// CHECK:     call i64 @pwrite({{.*}}4096,
// CHECK-NOT: 409600
NOINLINE void test_interleaved_io(int fd, int fd2, char *base, off_t off0) {
    for (int i = 0; i < N; i++) {
        pwrite(fd, base + (long)i * 4096, 4096, off0 + (long)i * 4096);
        write(fd2, base, 1);            // interleaved side-effecting I/O
    }
}

// ---------------------------------------------------------------------------
// NEGATIVE 4: runtime trip count. Tier-1 requires a constant backedge count
// (for the precise-range AA proof), so a symbolic 'n' must refuse.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_runtime_tripcount
// CHECK:     call i64 @pwrite({{.*}}4096,
// CHECK-NOT: 409600
NOINLINE void test_runtime_tripcount(int fd, char *base, off_t off0, int n) {
    for (int i = 0; i < n; i++)
        pwrite(fd, base + (long)i * 4096, 4096, off0 + (long)i * 4096);
}

