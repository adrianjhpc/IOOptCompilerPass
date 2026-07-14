// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-prefetch -S | %FileCheck %s --check-prefix=PREFETCH
// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s --check-prefix=NOPREFETCH

#include <unistd.h>

extern "C" {

// 32 sequential 4096-byte preads = 131072 bytes, above the 64KiB threshold.
// pread loops are handled by neither hoisting nor batching, so this is the
// canonical WILLNEED target: one hint for the whole range in the preheader.
// Prefetch is opt-in: PREFETCH requires -io-opt-prefetch; default is negative.

// PREFETCH-LABEL: define {{.*}}@prefetch_loop
// NOPREFETCH-LABEL: define {{.*}}@prefetch_loop
void prefetch_loop(int fd, char* buf) {
    // PREFETCH: call i32 @posix_fadvise(i32 {{.*}}, i64 {{.*}}, i64 131072, i32 {{.*}}3)
    // PREFETCH: call i64 @pread(
    // NOPREFETCH-NOT: posix_fadvise
    for (int i = 0; i < 32; i++) {
        pread(fd, buf, 4096, (long)i * 4096);
    }
}

}

