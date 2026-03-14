// RUN: %clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes="mem2reg,instcombine,io-opt" -S | %FileCheck %s

#include <stdio.h>

// CHECK-LABEL: @test_hoist
int test_hoist(FILE *fp, int x, int y) {
    char buf[20];

    // FIX: We must pre-compute the pointer offset so its LLVM IR instruction
    // (getelementptr) mathematically dominates the insertion point of the first fread.
    // Otherwise, the pass's strict Dominator Tree safety scanner will block the hoist!
    char* buf2 = buf + 10;

    // CHECK: call i64 @fread(ptr {{.*}}, i64 1, i64 20, ptr {{.*}})
    // CHECK: mul nsw i32
    // CHECK: sdiv i32
    fread(buf, 1, 10, fp);

    // The CPU bound math operations
    int z = x * y + 42;
    int w = z / 2;

    fread(buf2, 1, 10, fp);

    return w + buf[0];
}
