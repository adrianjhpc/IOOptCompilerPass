// RUN: clang-20 -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | opt-20 -load-pass-plugin=%S/../build/src/libIOOpt.so -passes="mem2reg,instcombine,io-opt" -S | FileCheck-20 %s

#include <unistd.h>

// -----------------------------------------------------------------------------
// POSITIVE TEST: Linear CFG (Unconditional Jumps)
// -----------------------------------------------------------------------------
// CHECK-LABEL: @test_cross_block_hoist
int test_cross_block_hoist(int fd, int x) {
    char buf[20];
    
    // As long as the read is hoisted above the math, we won!
    // CHECK: call i64 @read
    // CHECK: mul nsw i32
    
    int y = x * 2;
    goto next_block; 
    
next_block:
    y = y * 3;
    goto final_block; 
    
final_block:
    read(fd, buf, 20);
    return buf[0] + y;
}

// -----------------------------------------------------------------------------
// NEGATIVE TEST: Speculative CFG (Conditional Jumps)
// -----------------------------------------------------------------------------
// CHECK-LABEL: @test_no_speculative_hoist
int test_no_speculative_hoist(int fd, int cond) {
    char buf[20];
    
    // FileCheck asserts that the branch (the 'if' check) happens FIRST.
    // CHECK: br i1 %{{.*}}
    
    // And the read stays trapped safely inside the 'if' block!
    // CHECK: call i64 @read
    if (cond) {
        read(fd, buf, 20);
        return buf[0];
    }
    
    return 0;
}
