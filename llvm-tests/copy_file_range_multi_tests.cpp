// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-copy-file-range -io-opt-cfr-fallback=false -S | %FileCheck %s

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

#define NOINLINE __attribute__((noinline))

// CHECK-LABEL: define {{.*}}test_two_pairs
NOINLINE ssize_t test_two_pairs(int src, int dst) {
    char a[4096];
    char b[8192];
    // CHECK: call i64 @copy_file_range(i32 {{.*}}, ptr null, i32 {{.*}}, ptr null, i64 4096, i32 0)
    // CHECK: call i64 @copy_file_range(i32 {{.*}}, ptr null, i32 {{.*}}, ptr null, i64 8192, i32 0)
    // CHECK-NOT: call {{.*}} @read(
    // CHECK-NOT: call {{.*}} @write(
    ssize_t n1 = read(src, a, 4096);
    ssize_t w1 = write(dst, a, n1);
    ssize_t n2 = read(src, b, 8192);
    ssize_t w2 = write(dst, b, n2);
    return w1 + w2;
}

// CHECK-LABEL: define {{.*}}test_three_pairs
NOINLINE void test_three_pairs(int s1, int d1, int s2, int d2, int s3, int d3) {
    char a[1024];
    char b[2048];
    char c[4096];
    // CHECK: call i64 @copy_file_range(i32 {{.*}}, ptr null, i32 {{.*}}, ptr null, i64 1024, i32 0)
    // CHECK: call i64 @copy_file_range(i32 {{.*}}, ptr null, i32 {{.*}}, ptr null, i64 2048, i32 0)
    // CHECK: call i64 @copy_file_range(i32 {{.*}}, ptr null, i32 {{.*}}, ptr null, i64 4096, i32 0)
    // CHECK-NOT: call {{.*}} @read(
    // CHECK-NOT: call {{.*}} @write(
    ssize_t n1 = read(s1, a, 1024);
    write(d1, a, n1);
    ssize_t n2 = read(s2, b, 2048);
    write(d2, b, n2);
    ssize_t n3 = read(s3, c, 4096);
    write(d3, c, n3);
}

// Mixed: one transformable pair + one same-fd (rejected) pair.
// Exactly one copy_file_range; the same-fd read/write survive.
// CHECK-LABEL: define {{.*}}test_mixed_valid_invalid
NOINLINE ssize_t test_mixed_valid_invalid(int src, int dst, int self) {
    char good[4096];
    char bad[4096];
    // CHECK: call i64 @copy_file_range(i32 {{.*}}, ptr null, i32 {{.*}}, ptr null, i64 4096, i32 0)
    // CHECK: call {{.*}} @read(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}4096)
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}})
    // CHECK-NOT: call {{.*}}@copy_file_range
    ssize_t ng = read(src, good, 4096);
    ssize_t wg = write(dst, good, ng);
    ssize_t nb = read(self, bad, 4096);
    ssize_t wb = write(self, bad, nb);
    return wg + wb;
}

int main(void) {
    int src = open("/dev/zero", O_RDONLY);
    int dst = open("/dev/null", O_WRONLY);
    if (src < 0 || dst < 0) return 1;
    test_two_pairs(src, dst);
    test_three_pairs(src, dst, src, dst, src, dst);
    test_mixed_valid_invalid(src, dst, dst);
    close(src);
    close(dst);
    return 0;
}

