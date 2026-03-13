// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// CHECK-LABEL: define dso_local noundef zeroext i1 @_Z21test_short_write_trapiPcS_
bool test_short_write_trap(int fd, char* buf1, char* buf2) {
    // Because they are contiguous in fd, it should batch them.
    // 2 calls of 10 bytes = 20 bytes. Misses the Vectored threshold, hits ShadowBuffer.
    // CHECK: %shadow.buf = alloca [20 x i8]
    // CHECK: call i64 @write(i32 {{.*}}, ptr {{.*}}shadow.buf{{.*}}, i64 {{.*}}20)

    // The pass safely spoofs the return value, preserving POSIX error codes!
    // It creates a select instruction that forwards negative error codes, or returns 10 on success.
    // CHECK: %spoofed.posix.ret = select i1 %{{.*}}, i64 %{{.*}}, i64 10
    // CHECK: icmp sgt i64 %spoofed.posix.ret, 9

    ssize_t w1 = write(fd, buf1, 10);
    if (w1 < 10) return false;

    ssize_t w2 = write(fd, buf2, 10);
    if (w2 < 10) return false;

    return true;
}
