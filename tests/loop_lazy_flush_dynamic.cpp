// RUN: clang++-20 -O1 -fno-inline -emit-llvm -S -c %s -o - | opt-20 -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | FileCheck-20 %s

#include <unistd.h>

// CHECK-LABEL: define {{.*}}test_dynamic_loop
__attribute__((noinline))
void test_dynamic_loop(int fd, const char* buffer, int dynamic_count) {
    
    // The loop iterations are unknown (dynamic_count).
    // SCEV cannot determine the trip count.
    // The pass must not hoist the write. It must remain inside the loop.
    
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}4)
    
    #pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < dynamic_count; ++i) {
        write(fd, buffer, 4);
    }
}
