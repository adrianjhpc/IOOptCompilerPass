// RUN: clang++-20 -O2 -fno-inline -emit-llvm -S -c %s -o - | opt-20 -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | FileCheck-20 %s

#include <unistd.h>
#include <sys/types.h>

extern "C" {

// CHECK-LABEL: define {{.*}}optimised_pwrite
void optimised_pwrite(int fd, const char* b1, const char* b2, const char* b3, const char* b4, size_t len, off_t offset) {
    // By using 4 different pointers (b1, b2, b3, b4), we force 
    // the classifier to use 'pwritev' because the memory is scattered.
    
    ssize_t r1 = pwrite(fd, b1, len, offset);
    if (r1 < 0) return;

    ssize_t r2 = pwrite(fd, b2, len, offset + len);
    if (r2 < 0) return;

    ssize_t r3 = pwrite(fd, b3, len, offset + (len * 2));
    if (r3 < 0) return;

    ssize_t r4 = pwrite(fd, b4, len, offset + (len * 3));
    if (r4 < 0) return;

    // NOW the check will find the iovec array
    // CHECK: %iovec.array.N = alloca [4 x { ptr, i64 }], align 8
    // CHECK: call i64 @pwritev
}

}
