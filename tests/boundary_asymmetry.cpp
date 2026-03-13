// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// CHECK-LABEL: define {{.*}}test_asymmetric_reads
__attribute__((noinline))
void test_asymmetric_reads(int fd, char* b1, size_t l1, char* b2, size_t l2) {
    //Reads only need N>=2. This must vectorize.
    // CHECK: call {{.*}} @readv(i32 {{.*}}, ptr {{.*}}, i32 2)
    read(fd, b1, l1);
    read(fd, b2, l2);
}

// CHECK-LABEL: define {{.*}}test_asymmetric_writes
__attribute__((noinline))
void test_asymmetric_writes(int fd, const char* b1, size_t l1, const char* b2, size_t l2) {
    // Writes need N>=4 (dynamic sizes). This must not vectorize.
    // CHECK-NOT: call {{.*}} @writev
    // CHECK: call {{.*}} @write
    // CHECK: call {{.*}} @write
    write(fd, b1, l1);
    write(fd, b2, l2);
}
