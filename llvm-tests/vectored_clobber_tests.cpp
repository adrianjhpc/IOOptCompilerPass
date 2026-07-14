// RUN: %ppclang -O2 -fno-inline -fno-unroll-loops -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-loop-vectored -S | %FileCheck %s
//
// DVLC (vector loop collapse) no-clobber proof hardening.
//
// The earlier positive tests had NO producing stores, so the deferral-safety
// scan (the guard against silent corruption) never ran on a true fire. These
// exercise it directly:
//   * a per-slot producing store inside the buffer's own slot  -> MUST still fire
//   * a cross-slot store (writes a different slot)             -> MUST refuse
//   * a stride-mismatched store (walks the buffer differently) -> MUST refuse
//
// NOTE: producing stores are plain 'store' instructions, not memset/memcpy: the
// interleave scan rejects ANY non-readonly call (incl. mem intrinsics),
// so an intrinsic fill would refuse for an unrelated reason and wouldn't test
// the SCEV no-clobber logic. (That intrinsic limitation is real for actual code
// -- see the plugin notes.)
//
// The negative-offset branch (a store starting BEFORE the buffer base) is not
// tested here: such a store targets already-consumed earlier slots, is dead, and
// is removed by DCE before the pass runs -- so a *live* instance cannot exist in
// this straight-line shape. It is covered by construction in the scan.
//
// Detection: "writev(" implies both writev and pwritev (substring). Length/iovcnt
// are anchored on trailing chars so "4096)" and "i32 100)" are unambiguous.

#include <unistd.h>
#include <sys/types.h>

#define NOINLINE __attribute__((noinline))
#define N 100

// ---------------------------------------------------------------------------
// POSITIVE: each iteration stores into offset 0 of its OWN slot (Diff == 0,
// step == stride), then writes that slot. The no-clobber scan must APPROVE the
// store (own-slot) and the loop must collapse to writev.
// The store is live: write() reads the slot it just filled, so DCE keeps it.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_own_slot_store_fires
// CHECK:     call{{.*}}@writev({{.*}}i32 100)
// CHECK-NOT: call{{.*}}@write(
NOINLINE void test_own_slot_store_fires(int fd, char *base) {
    for (int i = 0; i < N; i++) {
        base[(long)i * 8192] = (char)i;          // own slot i, offset 0 (< count)
        write(fd, base + (long)i * 8192, 4096);  // stride 8192 > count 4096
    }
}

// ---------------------------------------------------------------------------
// NEGATIVE 1 (cross-slot): iteration i stores into slot i+1 (Diff == +stride,
// which is >= count), then writes slot i. The store is live (read by write() at
// iteration i+1), so it survives DCE. The scan must REFUSE via the
// "start-offset outside its own slot" branch (Diff >= count). No writev.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_cross_slot_store_refuses
// CHECK:     call{{.*}}@write({{.*}}4096)
// CHECK-NOT: call{{.*}}@writev(
// CHECK-NOT: call{{.*}}@pwritev(
NOINLINE void test_cross_slot_store_refuses(int fd, char *base) {
    for (int i = 0; i < N; i++) {
        base[(long)(i + 1) * 8192] = (char)i;    // slot i+1 (Diff = +8192 >= 4096)
        write(fd, base + (long)i * 8192, 4096);
    }
}

// ---------------------------------------------------------------------------
// NEGATIVE 2 (stride mismatch): a store that walks the buffer at 16384 while the
// write buffer strides by 8192. WStep (16384) != stride (8192), so the scan must
// REFUSE via the step-mismatch branch. The store is live (slot 2i is read by
// write() at iteration 2i), so DCE keeps it. No writev.
// ---------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_stride_mismatch_store_refuses
// CHECK:     call{{.*}}@write({{.*}}4096)
// CHECK-NOT: call{{.*}}@writev(
// CHECK-NOT: call{{.*}}@pwritev(
NOINLINE void test_stride_mismatch_store_refuses(int fd, char *base) {
    for (int i = 0; i < N; i++) {
        base[(long)i * 16384] = (char)i;         // stride 16384 != 8192
        write(fd, base + (long)i * 8192, 4096);
    }
}

