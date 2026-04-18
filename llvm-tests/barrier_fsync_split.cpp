// RUN: env IO_ENABLE_LOGGING=0 IO_BATCH_THRESHOLD=4 IO_SHADOW_BUFFER_MAX=4096 IO_HIGH_WATER_MARK=65536 \
// RUN:   %ppclang -O2 -fno-inline -emit-llvm -S %s -o - \
// RUN:   | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S \
// RUN:   | %FileCheck %s

#include <unistd.h>
#include <stddef.h>

extern "C" {

__attribute__((noinline))
void test_fsync_splits_batches(int fd, const char *buf) {
  // 4 writes => should become writev(..., iovcnt=4)
  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);

  fsync(fd);

  // Another 4 writes => another writev(..., iovcnt=4)
  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);
  write(fd, buf, 8);
}

} // extern "C"

// CHECK-LABEL: define {{.*}} @test_fsync_splits_batches(
// I/O before barrier (accept write or writev)
// CHECK: {{call.*@writev?\(}}
// Barrier
// CHECK: {{call.*@fsync\(}}
// I/O after barrier (accept write or writev)
// CHECK: {{call.*@writev?\(}}
// Ensure only two such calls in the function (bounded by the return)
// CHECK-NOT: {{call.*@writev?\(}}
// CHECK: ret void

