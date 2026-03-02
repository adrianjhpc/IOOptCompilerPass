// RUN: clang-20 -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | opt-20 -load-pass-plugin=%S/../build/src/libIOOpt.so -passes="mem2reg,instcombine,io-opt" -S | FileCheck-20 %s

#include <stdio.h>

// CHECK-LABEL: @test_negative_interference
void test_negative_interference(FILE *fp) {
    char buf[20] = "01234567890123456789";
    
    // CHECK: call i64 @fwrite(ptr {{.*}}, i64 noundef 1, i64 noundef 10, ptr {{.*}})
    fwrite(buf, 1, 10, fp);
    
    // Memory interference prevents amalgamation
    // CHECK: store i8 88
    buf[15] = 'X'; 
    
    // CHECK: call i64 @fwrite(ptr {{.*}}, i64 noundef 1, i64 noundef 10, ptr {{.*}})
    fwrite(buf + 10, 1, 10, fp);
}
