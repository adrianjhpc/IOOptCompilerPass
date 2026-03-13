// RUN: %ppclang -O1 -fno-inline -emit-llvm -S -c %s -o - | \
// RUN: env IO_HIGH_WATER_MARK=32768 %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | \
// RUN: %FileCheck %s
#include <unistd.h>

// CHECK-LABEL: define {{.*}}test_highwater_loop
__attribute__((noinline))
void test_highwater_loop(int fd, const char* buffer) {
    
    // 100,000 iterations * 1 byte = 100,000 bytes.
    // 100,000 > 65536 (IOHighWaterMark).
    // The pass must abort the loop hoisting to protect the OS VFS queue.
    // We should not see a single 100000 byte write.
    
    // CHECK-NOT: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 100000)
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}1)
    
    #pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 100000; ++i) {
        write(fd, buffer, 1);
    }
}
