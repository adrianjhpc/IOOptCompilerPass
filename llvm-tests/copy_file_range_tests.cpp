// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-copy-file-range -io-opt-cfr-fallback=false -S | %FileCheck %s

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

#define NOINLINE __attribute__((noinline))

extern void escape(void *p);
extern void barrier(void);

// CHECK-LABEL: define {{.*}}test_copy_basic
NOINLINE ssize_t test_copy_basic(int src, int dst) {
    char buf[65536];
    // CHECK: call i64 @copy_file_range(i32 {{.*}}, ptr null, i32 {{.*}}, ptr null, i64 65536, i32 0)
    // CHECK-NOT: call {{.*}} @read(
    // CHECK-NOT: call {{.*}} @write(
    ssize_t n = read(src, buf, 65536);
    return write(dst, buf, n);
}

// CHECK-LABEL: define {{.*}}test_copy_pread
NOINLINE ssize_t test_copy_pread(int src, int dst, off_t oin, off_t oout) {
    char buf[4096];
    // CHECK-DAG: %cfr.offin = alloca i64
    // CHECK-DAG: %cfr.offout = alloca i64
    // CHECK: store i64 {{.*}}, ptr %cfr.offin
    // CHECK: store i64 {{.*}}, ptr %cfr.offout
    // CHECK: call i64 @copy_file_range(i32 {{.*}}, ptr %cfr.offin, i32 {{.*}}, ptr %cfr.offout, i64 4096, i32 0)
    // CHECK-NOT: call {{.*}} @pread(
    // CHECK-NOT: call {{.*}} @pwrite(
    ssize_t n = pread(src, buf, 4096, oin);
    return pwrite(dst, buf, n, oout);
}

// NEGATIVE: src == dst. Overlapping same-file copy is UB.
// CHECK-LABEL: define {{.*}}test_same_fd
NOINLINE ssize_t test_same_fd(int fd) {
    char buf[4096];
    // CHECK-NOT: call {{.*}}@copy_file_range
    ssize_t n = read(fd, buf, 4096);
    return write(fd, buf, n);
}

// NEGATIVE: buffer observed after the write -> not a pure bounce buffer.
// CHECK-LABEL: define {{.*}}test_buffer_observed
NOINLINE int test_buffer_observed(int src, int dst) {
    char buf[4096];
    // CHECK-NOT: call {{.*}}@copy_file_range
    ssize_t n = read(src, buf, 4096);
    write(dst, buf, n);
    return buf[0];
}

// NEGATIVE: buffer escapes to an opaque callee.
// CHECK-LABEL: define {{.*}}test_escape
NOINLINE ssize_t test_escape(int src, int dst) {
    char buf[4096];
    // CHECK-NOT: call {{.*}}@copy_file_range
    ssize_t n = read(src, buf, 4096);
    ssize_t w = write(dst, buf, n);
    escape(buf);
    return w;
}

// NEGATIVE: write length (100) != read result and not assume-full-reads.
// CHECK-LABEL: define {{.*}}test_length_mismatch
NOINLINE ssize_t test_length_mismatch(int src, int dst) {
    char buf[4096];
    // CHECK-NOT: call {{.*}}@copy_file_range
    read(src, buf, 4096);
    return write(dst, buf, 100);
}

// NEGATIVE: unrelated I/O to a different fd/buffer between R and W.
// CHECK-LABEL: define {{.*}}test_interleaved_io
NOINLINE ssize_t test_interleaved_io(int src, int dst, int other, char *ob) {
    char buf[4096];
    // CHECK-NOT: call {{.*}}@copy_file_range
    ssize_t n = read(src, buf, 4096);
    write(other, ob, 16);
    return write(dst, buf, n);
}

// NEGATIVE: opaque, side-effecting call between R and W.
// CHECK-LABEL: define {{.*}}test_interleaved_opaque
NOINLINE ssize_t test_interleaved_opaque(int src, int dst) {
    char buf[4096];
    // CHECK-NOT: call {{.*}}@copy_file_range
    ssize_t n = read(src, buf, 4096);
    barrier();
    return write(dst, buf, n);
}

int main(void) {
    int src = open("/dev/zero", O_RDONLY);
    int dst = open("/dev/null", O_WRONLY);
    if (src < 0 || dst < 0) return 1;
    test_copy_basic(src, dst);
    test_copy_pread(src, dst, 0, 0);
    test_same_fd(dst);
    test_buffer_observed(src, dst);
    test_escape(src, dst);
    test_length_mismatch(src, dst);
    test_interleaved_opaque(src, dst);
    close(src);
    close(dst);
    return 0;
}

