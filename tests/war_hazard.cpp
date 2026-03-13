// RUN: %clang -O2 -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// Use a wildcard to match the mangled C++ name: _Z15test_war_hazardiPc
// CHECK-LABEL: define {{.*}}@{{.*}}test_war_hazard
void test_war_hazard(int fd, char* buf) {
    // The pass MUST NOT merge these. We check they appear in the original order.

    // 1. The first read
    // CHECK: call i64 @read(i32 noundef %0, ptr noundef %1, i64 noundef 10)

    // 2. The memory hazard (the store)
    // CHECK: store i8 88

    // 3. The second read must remain AFTER the store
    // CHECK: call i64 @read(i32 noundef %0, ptr noundef {{.*}}, i64 noundef 10)

    read(fd, buf, 10);
    buf[15] = 'X';
    read(fd, buf + 10, 10);
}
