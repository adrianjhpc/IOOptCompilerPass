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
void test_posix_fadvise_splits_batches(int fd, const char *buf, size_t len) {
  // Dynamic len prevents Strided/ShadowBuffer paths and forces writev (N>=4)
  write(fd, buf, len);
  write(fd, buf, len);
  write(fd, buf, len);
  write(fd, buf, len);

  posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);

  write(fd, buf, len);
  write(fd, buf, len);
  write(fd, buf, len);
  write(fd, buf, len);
}

__attribute__((noinline))
void test_madvise_flushes_all(int fd, const char *buf, size_t len) {
  write(fd, buf, len);
  write(fd, buf, len);
  write(fd, buf, len);
  write(fd, buf, len);

  madvise((void*)buf, 4096, MADV_DONTNEED);

  write(fd, buf, len);
  write(fd, buf, len);
  write(fd, buf, len);
  write(fd, buf, len);
}

} // extern "C"

// CHECK-LABEL: define {{.*}} @test_posix_fadvise_splits_batches(
// Ensure we do NOT get one writev spanning the barrier.
// CHECK-NOT: call{{.*}} @writev{{.*}} i32 8
// We expect: writev(4) ... posix_fadvise ... writev(4)
// CHECK: call{{.*}} @writev{{.*}} i32 4
// CHECK: call{{.*}} @posix_fadvise
// CHECK: call{{.*}} @writev{{.*}} i32 4

// CHECK-LABEL: define {{.*}} @test_madvise_flushes_all(
// CHECK-NOT: call{{.*}} @writev{{.*}} i32 8
// CHECK: call{{.*}} @writev{{.*}} i32 4
// CHECK: call{{.*}} @madvise
// CHECK: call{{.*}} @writev{{.*}} i32 4

