// RUN: %clang -O3 -flto=full -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <stdio.h>

static void my_log(FILE* f, const char* msg) {
    fwrite(msg, 1, 4, f);
}

static void app_trace(FILE* f) {
    my_log(f, "INFO");
}

// FIX: Use a wildcard to match the mangled C++ name
// CHECK-LABEL: define {{.*}}@{{.*}}test_ipa_collapse
void test_ipa_collapse(FILE* f) {
    // The pass should inline the wrappers and find two adjacent 4-byte fwrites.
    // Because they are constant and tiny, it should use the Strided SIMD engine.

    // CHECK: %simd.shadow.buf = alloca <2 x i32>
    // CHECK: %gather.insert = insertelement <2 x i32>
    
    // It should issue a single 8-byte write (2 elements * 4 bytes).
    // CHECK: call i64 @fwrite(ptr {{.*}}%simd.shadow.buf, i64 1, i64 8, ptr {{.*}})
    
    app_trace(f);
    my_log(f, "DATA");
}
