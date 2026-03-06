// RUN: clang++-20 -O2 -fno-inline -emit-llvm -S -c %s -o - | opt-20 -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | FileCheck-20 %s

#include <unistd.h>

// Use extern "C" to prevent C++ name mangling in the LLVM IR!
extern "C" void external_side_effect_func();

// CHECK-LABEL: define {{.*}}test_opaque_barrier
__attribute__((noinline))
void test_opaque_barrier(int fd, const char* str1, size_t len1, const char* str2, size_t len2) {
    
    // The pass must not merge these writes into a writev or shadow buffer.
    // The external function acts as an impenetrable wall.
    
    // CHECK-NOT: alloca [{{.*}} x { ptr, i64 }]
    // CHECK-NOT: call {{.*}} @writev
    
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}})
    // CHECK: call {{.*}} @external_side_effect_func()
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}})
    
    write(fd, str1, len1);
    external_side_effect_func();
    write(fd, str2, len2);
}
