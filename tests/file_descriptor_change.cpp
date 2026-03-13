// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// CHECK-LABEL: define {{.*}}test_fd_mutation
__attribute__((noinline))
void test_fd_mutation(int* fd_ptr, const char* buf1, const char* buf2) {
    
    // The file descriptor is loaded from memory (*fd_ptr). 
    // Both write calls load from the exact same memory address.
    // However, the memory is mutated (store i32 99) between the calls.
    // The pass must recognize the integer aliasing hazard and abort batching.
    
    // CHECK-NOT: call {{.*}} @writev
    
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}10)
    // CHECK: store i32 99, ptr {{.*}}
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}10)
    
    write(*fd_ptr, buf1, 10);
    
    *fd_ptr = 99; // THE HAZARD: The underlying integer is overwritten!
    
    write(*fd_ptr, buf2, 10);
}
