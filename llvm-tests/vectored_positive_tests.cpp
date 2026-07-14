// RUN: %ppclang -O2 -fno-inline -fno-unroll-loops -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-loop-vectored -S | %FileCheck %s
//
// Detection metric: a single vectored call whose iovcnt is i32 100 (== N),
// with the original per-iteration syscall gone.
//   * write  loop (implicit offset) -> writev  (no offset proof needed)
//   * pwrite loop (contiguous offset) -> pwritev (offset addrec step == count)
// The vectored call lands in the loop EXIT block (deferred write).

#include <unistd.h>
#include <sys/types.h>

#define NOINLINE __attribute__((noinline))
#define N 100

// POS 1: scattered write loop (stride 8192 > count 4096). writev writes iov
// entries sequentially, advancing the fd offset == N sequential writes, so no
// offset proof is required.
// CHECK-LABEL: define {{.*}}test_writev_collapse
// CHECK:     call{{.*}}@writev({{.*}}i32 100)
// CHECK-NOT: call{{.*}}@write(
NOINLINE void test_writev_collapse(int fd, char *base) {
    for (int i = 0; i < N; i++)
        write(fd, base + (long)i * 8192, 4096);
}

// POS 2: scattered pwrite loop, contiguous file offset (offset stride == count).
// One pwritev with iovcnt 100 and the invariant base offset.
// CHECK-LABEL: define {{.*}}test_pwritev_collapse
// CHECK:     call{{.*}}@pwritev({{.*}}i32 100,
// CHECK-NOT: call{{.*}}@pwrite(
NOINLINE void test_pwritev_collapse(int fd, char *base, off_t off0) {
    for (int i = 0; i < N; i++)
        pwrite(fd, base + (long)i * 8192, 4096, off0 + (long)i * 4096);
}

