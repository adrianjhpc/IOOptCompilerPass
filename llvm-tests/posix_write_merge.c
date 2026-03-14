// RUN: %clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes="mem2reg,instcombine,io-opt" -S | %FileCheck %s

#include <unistd.h>

// CHECK-LABEL: @test_posix_write
void test_posix_write(int fd) {
    char buf[20] = "01234567890123456789";
    
    // FileCheck asserts that we see ONE write of length 20 (Arg 2 for write).
    // Note: LLVM treats size_t as i64 on 64-bit platforms.
    // CHECK: call i64 @write(i32 {{.*}}, ptr {{.*}}, i64 20)
    
    // FileCheck asserts that we do not see another write after it.
    // CHECK-NOT: call i64 @write
    write(fd, buf, 10);
    write(fd, buf + 10, 10);
}
