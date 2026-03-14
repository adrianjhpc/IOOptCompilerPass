// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// CHECK-LABEL: define dso_local void @_Z19test_pwrite_overlapiPcS_
void test_pwrite_overlap(int fd, char* buf1, char* buf2) {
    // Should NOT be merged because the offsets are identical (both 0).
    // CHECK: call i64 @pwrite(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}10, i64 {{.*}}0)
    // CHECK: call i64 @pwrite(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}10, i64 {{.*}}0)
    // CHECK-NOT: pwritev
    // CHECK-NOT: shadow.buf
    
    pwrite(fd, buf1, 10, 0);
    pwrite(fd, buf2, 10, 0); // This intentionally overwrites buf1 on the disk.
}
