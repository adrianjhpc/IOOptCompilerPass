// RUN: clang-20 -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | opt-20 -load-pass-plugin=%S/../build/src/libIOOpt.so -passes="mem2reg,instcombine,io-opt" -S | FileCheck-20 %s

#include <stdio.h>

// CHECK-LABEL: @test_hoist
int test_hoist(FILE *fp, int x, int y) {
    char buf[20];
    
    // The CPU bound math operations
    int z = x * y + 42;
    int w = z / 2;
    
    // In the C code, fread happens after the math
    // But our pass should hoist it to happen before the math in the IR
    
    // CHECK: call i64 @fread
    // CHECK: mul nsw i32
    // CHECK: sdiv i32
    fread(buf, 1, 20, fp);
    
    return w + buf[0];
}
