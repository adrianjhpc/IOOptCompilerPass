// RUN: %clang -O2 -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// Use a very loose label match to bypass C++ name mangling
// CHECK-LABEL: test_length_dominance_hazard
void test_length_dominance_hazard(int fd, char* buf, int x) {
    // VERIFICATION:
    // The pass MUST NOT merge these because the length for the second read
    // is calculated between the two calls.
    // The pass correctly detects that 'dynamic_len' does not dominate the first call.

    // 1. Match the first read (handling potential 'tail' keyword)
    // CHECK: call i64 {{.*}}@read(i32 noundef {{.*}}, ptr noundef {{.*}}, i64 noundef 10)

    // 2. Match the calculation that exists between them
    // CHECK: shl nsw i32

    // 3. Match the second read (proving it was NOT hoisted/merged)
    // CHECK: call i64 {{.*}}@read(i32 noundef {{.*}}, ptr noundef {{.*}}, i64 noundef {{.*}})

    read(fd, buf, 10);
    int dynamic_len = x * 2;
    read(fd, buf + 10, dynamic_len);
}
