// RUN: %clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes="mem2reg,instcombine,io-opt" -S | %FileCheck %s

#include <unistd.h>

// CHECK-LABEL: @test_posix_read_hoist
int test_posix_read_hoist(int fd, int x) {
    char buf[20];

    // We split the read. The pass will merge these into a single 20-byte read
    // and hoist the result to the location of this first call!
    
    // CHECK: call i64 @read(i32 {{.*}}, ptr {{.*}}, i64 20)
    // CHECK: mul nsw i32
    read(fd, buf, 10);

    int y = x * 42;
    int z = y / 2;

    read(fd, buf + 10, 10);

    return z + buf[0];
}
