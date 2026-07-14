// With -io-opt-cfr-assume-full-reads, a constant write length equal to the read
// count is accepted (the "ignore the return, assume a full read" idiom).
// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-copy-file-range -io-opt-cfr-assume-full-reads -io-opt-cfr-fallback=false -S | %FileCheck %s
//
// Without that flag the SAME input must be left untouched.
// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-copy-file-range -io-opt-cfr-fallback=false -S | %FileCheck %s --check-prefix=NOFLAG

#define _GNU_SOURCE
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

#define NOINLINE __attribute__((noinline))

// CHECK-LABEL: define {{.*}}test_full_read_copy
// NOFLAG-LABEL: define {{.*}}test_full_read_copy
NOINLINE ssize_t test_full_read_copy(int src, int dst) {
    char buf[4096];
    // read return value is ignored; both lengths are the constant 4096.
    // CHECK: call i64 @copy_file_range(i32 {{.*}}, ptr null, i32 {{.*}}, ptr null, i64 4096, i32 0)
    // NOFLAG-NOT: copy_file_range
    read(src, buf, 4096);
    return write(dst, buf, 4096);
}

int main(void) {
    int src = open("/dev/zero", O_RDONLY);
    int dst = open("/dev/null", O_WRONLY);
    if (src < 0 || dst < 0) return 1;
    test_full_read_copy(src, dst);
    close(src);
    close(dst);
    return 0;
}

