// RUN: %ppclang -O2 -fno-inline -fno-unroll-loops -emit-llvm -S %s -o - \
// RUN:   | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes="loop-simplify,lcssa,io-opt" -S \
// RUN:   | %FileCheck %s

#include <unistd.h>
#include <stddef.h>

extern "C" {

// Unsafe: conditional clobber does NOT dominate the write.
// Expect: no hoist (still a loop with write(...,16)), and no write(...,64).

// CHECK-LABEL: define {{.*}} @test_hoist_unsafe_conditional_clobber(
__attribute__((noinline))
void test_hoist_unsafe_conditional_clobber(int fd, char *buf) {
  for (int i = 0; i < 4; i++) {
    if (i & 1) {
      buf[i * 16 + 8] = (char)i;
    }
    write(fd, buf + i * 16, 16);
  }
}

// Still one write of 16 bytes should remain in the loop.
// CHECK:     call{{.*}} @write{{(64)?}}{{.*}} i64{{.*}}16
// Must not hoist into a single 64-byte write.
// CHECK-NOT: call{{.*}} @write{{(64)?}}{{.*}} i64{{.*}}64

} // extern "C"

