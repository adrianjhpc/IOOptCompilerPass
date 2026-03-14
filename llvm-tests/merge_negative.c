// RUN: %clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes="mem2reg,instcombine,io-opt" -S | %FileCheck %s

#include <stdio.h>

// CHECK-LABEL: @test_negative_interference
void test_negative_interference(FILE *fp) {
    char buf[20] = "01234567890123456789";

    // CHECK: call i64 @fwrite(ptr {{.*}}, i64 noundef 1, i64 noundef 10, ptr {{.*}})
    fwrite(buf, 1, 10, fp);

    // FIX: We must modify an index inside the FIRST 10 bytes (e.g., buf[5]) to create a TRUE 
    // memory overlap hazard. The compiler's Alias Analysis will detect this and safely abort.
    // CHECK: store i8 88
    buf[5] = 'X';

    // CHECK: call i64 @fwrite(ptr {{.*}}, i64 noundef 1, i64 noundef 10, ptr {{.*}})
    fwrite(buf + 10, 1, 10, fp);
}
