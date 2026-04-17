// RUN: env IO_ENABLE_LOGGING=0 IO_BATCH_THRESHOLD=4 IO_SHADOW_BUFFER_MAX=4096 IO_HIGH_WATER_MARK=1048576 \
// RUN:   %ppclang -O2 -fno-inline -emit-llvm -S %s -o - \
// RUN:   | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S \
// RUN:   | %FileCheck %s

#include <unistd.h>
#include <stddef.h>

extern "C" {

__attribute__((noinline))
void test_iov_max_writev_split(int fd, const char *buf) {
  // We intentionally generate >1024 adjacent writes so a single writev would exceed IOV_MAX.
  // Each write is length 8, and we use the same buffer pointer so the pass cannot classify
  // the memory as contiguous (Buf1+Len != Buf2), forcing writev rather than a single write.

#define W  write(fd, buf, 8);

#define REP2(X)   X X
#define REP4(X)   REP2(X) REP2(X)
#define REP8(X)   REP4(X) REP4(X)
#define REP16(X)  REP8(X) REP8(X)
#define REP32(X)  REP16(X) REP16(X)
#define REP64(X)  REP32(X) REP32(X)
#define REP128(X) REP64(X) REP64(X)
#define REP256(X) REP128(X) REP128(X)
#define REP512(X) REP256(X) REP256(X)
#define REP1024(X) REP512(X) REP512(X)

  // Total writes = 1024 + 26 = 1050
  REP1024(W)
  REP16(W) REP8(W) REP2(W)

#undef REP1024
#undef REP512
#undef REP256
#undef REP128
#undef REP64
#undef REP32
#undef REP16
#undef REP8
#undef REP4
#undef REP2
#undef W
}

} // extern "C"

// CHECK-LABEL: define {{.*}} @test_iov_max_writev_split(
// The pass should not emit a single oversized writev with iovcnt 1050.
// CHECK-NOT: call{{.*}} @writev{{.*}} i32 1050

// With IOV_MAX capping implemented (typically 1024), we expect at least two writev calls:
// one with iovcnt 1024 and another with iovcnt 26.
// CHECK: call{{.*}} @writev{{.*}} i32 1024
// CHECK: call{{.*}} @writev{{.*}} i32 26

// And no scalar writes should remain in this function.
// CHECK-NOT: call{{.*}} @write(

