// The transform is opt-in: absent -io-opt-copy-file-range, nothing happens.
// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s
//
// Even opted-in, it is suppressed when we cannot assume regular files
// (copy_file_range is meaningless on pipes/sockets).
// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-copy-file-range -io-opt-assume-regular-files=false -S | %FileCheck %s

#define _GNU_SOURCE
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

#define NOINLINE __attribute__((noinline))

// CHECK-LABEL: define {{.*}}test_gated
NOINLINE ssize_t test_gated(int src, int dst) {
    char buf[65536];
    // CHECK-NOT: copy_file_range
    ssize_t n = read(src, buf, 65536);
    return write(dst, buf, n);
}

int main(void) {
    int src = open("/dev/zero", O_RDONLY);
    int dst = open("/dev/null", O_WRONLY);
    if (src < 0 || dst < 0) return 1;
    test_gated(src, dst);
    close(src);
    close(dst);
    return 0;
}

