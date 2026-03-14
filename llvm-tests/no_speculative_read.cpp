// RUN: %clang -O2 -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// Loose label match to ignore C++ name mangling
// CHECK-LABEL: test_no_speculative_read
void test_no_speculative_read(int fd, char* buf, int cond) {
    // VERIFICATION:
    // The read is inside an 'if' block.
    // The pass must NOT hoist it because the block does not post-dominate the entry.
    // Speculative I/O is a major safety hazard and must be blocked.

    // 1. Match the conditional branch
    // CHECK: br i1 {{.*}}, label %{{.*}}, label %{{.*}}

    // 2. Match the read inside its original block (allowing for 'tail' keyword)
    // CHECK: call i64 {{.*}}@read(i32 noundef {{.*}}, ptr noundef {{.*}}, i64 noundef 10)

    if (cond) {
        read(fd, buf, 10);
    }
}
