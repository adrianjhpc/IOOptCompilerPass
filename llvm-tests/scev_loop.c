// RUN: %clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes="mem2reg,instcombine,loop-simplify,loop-rotate,indvars,io-opt" -S | %FileCheck %s

#include <unistd.h>

// CHECK-LABEL: @test_scev_loop
void test_scev_loop(int fd) {
    char buf[1000];
    
    // FileCheck asserts that the write happens exactly once, 
    // outside the loop, with a total length of 100 * 10 = 1000!
    // CHECK: call i64 @write(i32 {{.*}}, ptr {{.*}}, i64 1000)
    
    // The loop body should contain no write calls.
    // CHECK-NOT: call i64 @write
    
    for (int i = 0; i < 100; i++) {
        write(fd, buf + (i * 10), 10);
    }
}
