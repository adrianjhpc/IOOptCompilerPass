// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// CHECK-LABEL: define {{.*}}test_shadow_buffer
__attribute__((noinline))
void test_shadow_buffer(int fd) {
    char header[10] = "Header...";
    char footer[10] = "Footer...";

    // It should allocate a 20-byte array on the stack (10 + 10)
    // CHECK: %shadow.buf = alloca [20 x i8]
    
    // It should use llvm.memcpy to pack the scattered buffers
    // CHECK: call void @llvm.memcpy
    // CHECK: call void @llvm.memcpy
    
    // It should issue a single write of 20 bytes
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}%shadow.buf, i64 20)
    
    write(fd, header, 10);
    write(fd, footer, 10);
}
