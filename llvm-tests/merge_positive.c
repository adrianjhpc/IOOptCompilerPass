// RUN: %clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes="mem2reg,instcombine,io-opt" -S | %FileCheck %s
#include <stdio.h>

// CHECK-LABEL: @test_positive_merge
void test_positive_merge(FILE *fp) {
    char buf[20] = "01234567890123456789";
    
    // FileCheck asserts that we see ONE fwrite of length 20.
    // CHECK: call i64 @fwrite(ptr %{{.*}}, i64 1, i64 20, ptr %{{.*}})
    
    // FileCheck asserts that we do not see another fwrite after it.
    // CHECK-NOT: call i64 @fwrite
    fwrite(buf, 1, 10, fp);
    fwrite(buf + 10, 1, 10, fp);
}
