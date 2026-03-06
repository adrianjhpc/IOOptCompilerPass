// RUN: clang++-20 -O1 -fno-inline -emit-llvm -S -c %s -o - | opt-20 -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | FileCheck-20 %s

#include <unistd.h>

// CHECK-LABEL: define {{.*}}test_static_loop
__attribute__((noinline))
void test_static_loop(int fd, const char* buffer) {
    
    // The pass not remove the write from inside the loop body.
    // It must multiply 10 iterations * 4 bytes = 40 bytes.
    // It must place a single 40-byte write at the loop exit block.
    
    // CHECK-NOT: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 4)
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 40)
    
    #pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 10; ++i) {
        write(fd, buffer, 4);
    }
}
