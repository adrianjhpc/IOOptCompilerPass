// RUN: %ppclang -O2 -fno-inline -fno-unroll-loops -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=loop-simplify,lcssa,io-opt -io-opt-loop-hoist-dynamic-trips -S | %FileCheck %s
//
// Tier-1 explicit-offset loop collapse with a RUNTIME (symbolic) trip count.
//
// Runtime N forces clang to emit an `if (n>0)` guard whose false edge lands on
// the loop's exit block -> a NON-dedicated exit -> the loop is not in
// LoopSimplify form under bare `-passes=io-opt`, so the pass declines. The real
// optimizer pipelines prepend LoopSimplify+LCSSA; this RUN line does the same so
// a dedicated loop-exit exists (which is ALSO what makes the runtime-N write
// correct: the merged call sits on the loop-TAKEN path, and the n<=0 path
// bypasses it).
//
// Detection: plugin-created merged calls carry NO `noundef` (IRBuilder adds no
// param attrs), while clang's per-iteration calls do ("i32 noundef %0"). So:
//   * FIRED  -> a "@pwrite(i32 %<reg>, ptr ..." call (no noundef) exists, and no
//              "i32 noundef" per-iteration call survives.
//   * NOT    -> the per-iteration "i32 noundef ... 4096," call survives.
// (We do NOT test "no literal 4096": the merged length is 4096*N, which expands
// to a `mul ..., 4096`, legitimately leaving a 4096 in the function.)

#include <unistd.h>
#include <sys/types.h>

#define NOINLINE __attribute__((noinline))

// ---------------------------------------------------------------------------
// POSITIVE 1: runtime-N contiguous pwrite -> one pwrite of (4096*N) in the
// dedicated loop-exit. CHECK-NOT (before the match) rules out a leftover
// per-iteration call in the loop body; CHECK matches the merged (no-noundef)
// call that follows the loop.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_dynamic_pwrite_fires
// CHECK-NOT: call{{.*}}@pwrite(i32 noundef
// CHECK:     call{{.*}}@pwrite(i32 %{{[0-9]+}}, ptr
NOINLINE void test_dynamic_pwrite_fires(int fd, char *base, off_t off0, int n) {
    for (int i = 0; i < n; i++)
        pwrite(fd, base + (long)i * 4096, 4096, off0 + (long)i * 4096);
}

// ---------------------------------------------------------------------------
// POSITIVE 2: runtime-N contiguous pread -> one pread of (4096*N) in the
// preheader (before the loop). CHECK matches the merged call; CHECK-NOT (after)
// rules out a leftover per-iteration call in the loop body.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_dynamic_pread_fires
// CHECK:     call{{.*}}@pread(i32 %{{[0-9]+}}, ptr
// CHECK-NOT: call{{.*}}@pread(i32 noundef
NOINLINE void test_dynamic_pread_fires(int fd, char *base, off_t off0, int n) {
    for (int i = 0; i < n; i++)
        pread(fd, base + (long)i * 4096, 4096, off0 + (long)i * 4096);
}

// ---------------------------------------------------------------------------
// NEGATIVE 1: offset stride (8192) != count (4096). Offset-addrec proof fails
// even with runtime N; the per-iteration literal-4096 call survives.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_dynamic_noncontiguous_refuses
// CHECK: call{{.*}}@pwrite({{.*}}4096,
NOINLINE void test_dynamic_noncontiguous_refuses(int fd, char *base, off_t off0, int n) {
    for (int i = 0; i < n; i++)
        pwrite(fd, base + (long)i * 4096, 4096, off0 + (long)i * 8192);
}

// ---------------------------------------------------------------------------
// NEGATIVE 2: return value used -> use_empty()/LCSSA gate refuses; per-iteration
// call survives.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_dynamic_used_return_refuses
// CHECK: call{{.*}}@pwrite({{.*}}4096,
NOINLINE ssize_t test_dynamic_used_return_refuses(int fd, char *base, off_t off0, int n) {
    ssize_t total = 0;
    for (int i = 0; i < n; i++)
        total += pwrite(fd, base + (long)i * 4096, 4096, off0 + (long)i * 4096);
    return total;
}

// ---------------------------------------------------------------------------
// NEGATIVE 3: interleaved I/O on another fd -> side-effects scan refuses (the
// flag relaxes ONLY the trip-count requirement); per-iteration call survives.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_dynamic_interleaved_refuses
// CHECK: call{{.*}}@pwrite({{.*}}4096,
NOINLINE void test_dynamic_interleaved_refuses(int fd, int fd2, char *base, off_t off0, int n) {
    for (int i = 0; i < n; i++) {
        pwrite(fd, base + (long)i * 4096, 4096, off0 + (long)i * 4096);
        write(fd2, base, 1);
    }
}

