// RUN: %clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes="mem2reg,instcombine,io-opt" -S | %FileCheck %s

#include <unistd.h>

// -----------------------------------------------------------------------------
// Positive test: Cross-Block Batching & Hoisting
// -----------------------------------------------------------------------------
// CHECK-LABEL: @test_cross_block_hoist
int test_cross_block_hoist(int fd, int x) {
    char buf[20];
    
    // FIX: Pre-compute the pointer so the GEP instruction is placed in the 
    // entry block. This allows it to safely dominate the hoisted read!
    char* buf2 = buf + 10;

    // We place one read in the entry block, and one in the final block.
    // The pass must mathematically prove they are contiguous, merge them into 
    // a single 20-byte read, and HOIST the merged call to the location of the first read.
    
    // CHECK: call i64 @read(i32 {{.*}}, ptr {{.*}}, i64 20)
    // CHECK: mul nsw i32

    read(fd, buf, 10); // First half

    int y = x * 2;
    goto next_block;

next_block:
    y = y * 3;
    goto final_block;

final_block:
    read(fd, buf2, 10); // Second half
    return buf[0] + y;
}

// -----------------------------------------------------------------------------
// Negative test: Speculative CFG (conditional jumps)
// -----------------------------------------------------------------------------
// CHECK-LABEL: @test_no_speculative_hoist
int test_no_speculative_hoist(int fd, int cond) {
    char buf[20];

    // FileCheck asserts that the branch (the 'if' check) happens first
    // CHECK: br i1 %{{.*}}

    // And the read stays trapped safely inside the 'if' block
    // CHECK: call i64 @read
    if (cond) {
        read(fd, buf, 20);
        return buf[0];
    }

    return 0;
}
