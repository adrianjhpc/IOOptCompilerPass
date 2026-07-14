// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-copy-file-range -S | %FileCheck %s

#define _GNU_SOURCE
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

#define NOINLINE __attribute__((noinline))

// Default mode wraps copy_file_range with an ENOSYS(38)/EXDEV(18) fallback that
// re-runs the original read+write, merging both return values with phis.
//
// This also pins the branch DIRECTION: when a fallback is needed the guard must
// jump to %cfr.fallback (regression guard for the inverted-branch bug).
//
// CHECK-LABEL: define {{.*}}test_copy_with_fallback
NOINLINE ssize_t test_copy_with_fallback(int src, int dst) {
    char buf[65536];
    // CHECK: %cfr.ret = call i64 @copy_file_range(
    // CHECK: call ptr @__errno_location()
    // CHECK: icmp eq i32 {{.*}}, 38
    // CHECK: icmp eq i32 {{.*}}, 18
    // CHECK: br i1 %cfr.need.fb, label %cfr.fallback, label %cfr.cont
    // CHECK: cfr.fallback:
    // CHECK:   call {{.*}} @read(
    // CHECK:   call {{.*}} @write(
    // CHECK:   br label %cfr.cont
    // CHECK: cfr.cont:
    // CHECK:   phi i64
    // CHECK:   phi i64
    ssize_t n = read(src, buf, 65536);
    return write(dst, buf, n);
}

// Fallback mode must fold the pair EXACTLY ONCE and terminate (regression guard
// for the re-scan re-folding its own synthesised fallback read/write, which
// previously hung and then produced a broken PHI).
// CHECK-LABEL: define {{.*}}test_fallback_terminates
// CHECK-COUNT-1: call i64 @copy_file_range(
// CHECK-NOT:     call i64 @copy_file_range(
NOINLINE ssize_t test_fallback_terminates(int src, int dst) {
    char buf[65536];
    ssize_t n = read(src, buf, 65536);
    return write(dst, buf, n);
}

int main(void) {
    int src = open("/dev/zero", O_RDONLY);
    int dst = open("/dev/null", O_WRONLY);
    if (src < 0 || dst < 0) return 1;
    test_copy_with_fallback(src, dst);
    close(src);
    close(dst);
    return 0;
}

