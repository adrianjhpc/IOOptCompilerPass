// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// CHECK-LABEL: define {{.*}}test_vectored_dynamic
__attribute__((noinline))
void test_vectored_dynamic(int fd, const char* p1, size_t l1, const char* p2, size_t l2, const char* p3, size_t l3, const char* p4, size_t l4) {

    // Sizes are dynamic, so no shadow buffering.
    // But N=4 hits the strict profitability threshold.
    // It should build an iovec array of size 4 and call writev.

    // CHECK: %iovec.array.N = alloca [4 x { ptr, i64 }]
    // FIX: Update the CHECK line to look for %iovec.base.ptr instead of the alloca directly
    // CHECK: call {{.*}} @writev(i32 {{.*}}, ptr %iovec.base.ptr, i32 4)

    write(fd, p1, l1);
    write(fd, p2, l2);
    write(fd, p3, l3);
    write(fd, p4, l4);
}
