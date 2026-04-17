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

  // Should flush current batch for this fd
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

  // Your pass flushes all outstanding batches when it sees madvise()
  madvise((void*)buf, 4096, MADV_DONTNEED);

  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);
}

} // extern "C"

// CHECK-LABEL: define {{.*}} @test_posix_fadvise_splits_batches(
// CHECK-NOT: call{{.*}} @writev{{.*}} i32 8
// CHECK: call{{.*}} @writev{{.*}} i32 4
// CHECK: call{{.*}} @posix_fadvise
// CHECK: call{{.*}} @writev{{.*}} i32 4

// CHECK-LABEL: define {{.*}} @test_madvise_flushes_all(
// CHECK-NOT: call{{.*}} @writev{{.*}} i32 8
// CHECK: call{{.*}} @writev{{.*}} i32 4
// CHECK: call{{.*}} @madvise
// CHECK: call{{.*}} @writev{{.*}} i32 4

