// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// A standard struct. Each Point is 8 bytes total.
struct Point {
    int x; // 4 bytes
    int y; // 4 bytes
};

// CHECK-LABEL: define {{.*}}test_strided_gather
__attribute__((noinline))
void test_strided_gather(int fd, Point* pts) {
    
    // The pass should recognize 4 writes of exactly 4 bytes each.
    // It should allocate a 4-element SIMD vector on the stack.
    // CHECK: %simd.shadow.buf = alloca <4 x i32>
    
    // It should load the integers and insert them into the vector register.
    // CHECK: %strided.load = load i32
    // CHECK: %gather.insert = insertelement <4 x i32>
    
    // It should dump the entire vector register to the stack buffer in one go.
    // CHECK: store <4 x i32> %{{.*}}, ptr %simd.shadow.buf
    
    // It should issue a single write of 16 bytes (4 elements * 4 bytes).
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}%simd.shadow.buf, i64 16)
    
    write(fd, &pts[0].x, 4);
    write(fd, &pts[1].x, 4);
    write(fd, &pts[2].x, 4);
    write(fd, &pts[3].x, 4);
}
