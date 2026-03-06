// RUN: clang-20 -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | opt-20 -load-pass-plugin=%S/../build/src/libIOOpt.so -passes="mem2reg,instcombine,io-opt" -S | FileCheck-20 %s

#include <unistd.h>

// CHECK-LABEL: @test_posix_read_hoist
int test_posix_read_hoist(int fd, int x) {
    char buf[20];
    
    int y = x * 42;
    int z = y / 2;
    
    // FileCheck asserts that the read happens before the math
    // CHECK: call i64 @read
    // CHECK: mul nsw i32
    read(fd, buf, 20);
    
    return z + buf[0];
}
