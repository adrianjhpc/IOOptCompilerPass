// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// CHECK-LABEL: define {{.*}}test_unprofitable
__attribute__((noinline))
void test_unprofitable(int fd, const char* str1, size_t len1, const char* str2, size_t len2) {
    
    // N=2 is below the dynamic threshold (4).
    // The sizes are unknown, so it CANNOT use a ShadowBuffer.
    // The pass MUST abort and leave exactly two separate write calls.
    
    // CHECK-NOT: alloca [{{.*}} x { ptr, i64 }]
    // CHECK-NOT: call {{.*}} @writev
    
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}})
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}})
    
    write(fd, str1, len1);
    write(fd, str2, len2);
}
