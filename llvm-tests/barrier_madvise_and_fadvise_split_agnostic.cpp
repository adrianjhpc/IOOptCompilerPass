// RUN: env IO_ENABLE_LOGGING=0 IO_BATCH_THRESHOLD=4 IO_SHADOW_BUFFER_MAX=4096 IO_HIGH_WATER_MARK=65536 \
// RUN:   %ppclang -O2 -fno-inline -emit-llvm -S %s -o - \
// RUN:   | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S \
// RUN:   | %FileCheck %s

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stddef.h>

extern "C" {

__attribute__((noinline))
void test_posix_fadvise_splits_batches(int fd, const char *buf) {
  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);

  posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);

  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);
}

__attribute__((noinline))
void test_madvise_flushes_all(int fd, const char *buf) {
  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);

  madvise((void*)buf, 4096, MADV_DONTNEED);

  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);
}

} // extern "C"

// We accept either writev() or a single coalesced write() (e.g., Strided/ShadowBuffer).
// Match either "@write" or "@writev" using the optional 'v' regex.
// (This intentionally does NOT constrain the byte count or iovcnt.)
// CHECK-LABEL: define {{.*}} @test_posix_fadvise_splits_batches(
// CHECK: call{{.*}} @write{{v?}}(
// CHECK: call{{.*}} @posix_fadvise
// CHECK: call{{.*}} @write{{v?}}(
// Ensure batching happened: no third write/writev call in the function.
// CHECK-NOT: call{{.*}} @write{{v?}}(
// CHECK: ret void

// CHECK-LABEL: define {{.*}} @test_madvise_flushes_all(
// CHECK: call{{.*}} @write{{v?}}(
// CHECK: call{{.*}} @madvise
// CHECK: call{{.*}} @write{{v?}}(
// Ensure batching happened: no third write/writev call in the function.
// CHECK-NOT: call{{.*}} @write{{v?}}(
// CHECK: ret void

