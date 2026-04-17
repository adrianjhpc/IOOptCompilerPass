// RUN: %ppclang -O2 -fno-inline -fno-unroll-loops -emit-llvm -S %s -o - \
// RUN:   | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes="loop-simplify,lcssa,io-opt" -S \
// RUN:   | %FileCheck %s

#include <unistd.h>
#include <stddef.h>

extern "C" {

//===----------------------------------------------------------------------===//
// Safe loop hoisting: slice-confined store dominates the write.
// Expect: hoisted single write(..., 64) and no write(..., 16) in this function.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define {{.*}} @test_hoist_safe_write(
__attribute__((noinline))
void test_hoist_safe_write(int fd, char *buf) {
  for (int i = 0; i < 4; i++) {
    buf[i * 16 + 8] = (char)i;
    write(fd, buf + i * 16, 16);
  }
}

// No 16-byte write anywhere before the hoisted write.
// CHECK-NOT: call{{.*}} @write{{(64)?}}{{.*}} i64{{.*}}16
// Hoisted call should be a single 64-byte write.
// CHECK:     call{{.*}} @write{{(64)?}}{{.*}} i64{{.*}}64
// After hoisting, no additional write calls should remain in this function.
// (Bounded by the next CHECK-LABEL below.)
// CHECK-NOT: call{{.*}} @write{{(64)?}}

//===----------------------------------------------------------------------===//
// Unsafe: store AFTER the write -> does not dominate -> must NOT hoist.
// Expect: still a loop with a write(...,16), and no write(...,64).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define {{.*}} @test_hoist_unsafe_poststore(
__attribute__((noinline))
void test_hoist_unsafe_poststore(int fd, char *buf) {
  for (int i = 0; i < 4; i++) {
    write(fd, buf + i * 16, 16);
    buf[i * 16 + 8] = (char)i;
  }
}

// CHECK:     call{{.*}} @write{{(64)?}}{{.*}} i64{{.*}}16
// CHECK-NOT: call{{.*}} @write{{(64)?}}{{.*}} i64{{.*}}64

//===----------------------------------------------------------------------===//
// Unsafe: loop-variant clobber outside the slice family (buf[0] each iter).
// Expect: still a loop with a write(...,16), and no write(...,64).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: define {{.*}} @test_hoist_unsafe_invariant_clobber(
__attribute__((noinline))
void test_hoist_unsafe_invariant_clobber(int fd, char *buf) {
  for (int i = 0; i < 4; i++) {
    buf[0] = (char)i;
    write(fd, buf + i * 16, 16);
  }
}

// CHECK:     call{{.*}} @write{{(64)?}}{{.*}} i64{{.*}}16
// CHECK-NOT: call{{.*}} @write{{(64)?}}{{.*}} i64{{.*}}64

} // extern "C"

