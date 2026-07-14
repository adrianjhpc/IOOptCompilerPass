// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-prefetch -S | %FileCheck %s --check-prefix=RANDOM
// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-prefetch -io-opt-prefetch-random=false -S | %FileCheck %s --check-prefix=NORANDOM

#include <unistd.h>
extern "C" {
// Large-strided pread: 512-byte reads spaced 64 KiB apart. Forward readahead
// would fault in ~63.5 KiB of unused pages per iteration -> advise RANDOM to
// suppress it (advice value 1), NOT SEQUENTIAL.

// RANDOM-LABEL: define {{.*}}@strided_reads
// NORANDOM-LABEL: define {{.*}}@strided_reads
void strided_reads(int fd, char* buf) {
    // RANDOM: call i32 @posix_fadvise(i32 {{.*}}, i64 0, i64 0, i32 1)
    // RANDOM-NOT: i32 2)
    // NORANDOM-NOT: posix_fadvise
    for (int i = 0; i < 1000; i++) {
        pread(fd, buf, 512, (long)i * 65536);
    }
}
}

