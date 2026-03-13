// RUN: %ppclang -O1 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes="loop-simplify,lcssa,io-opt" -S | %FileCheck %s

#include <unistd.h>

// CHECK-LABEL: define {{.*}}test_static_loop
__attribute__((noinline))
void test_static_loop(int fd, const char* buffer) {

    // The pass must remove the write from inside the loop body.
    // It must multiply 10 iterations * 4 bytes = 40 bytes.
    // It must place a single 40-byte write at the loop exit block.

    // CHECK-NOT: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 4)
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 40)

    #pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 10; ++i) {
        // FIX: We must increment the pointer by the exact size of the write (4)
        // so the compiler mathematically proves it is a contiguous array scan!
        write(fd, buffer + (i * 4), 4);
    }
}
