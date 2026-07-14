// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s --check-prefix=COALESCE
// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -enable-io-opt=false -S | %FileCheck %s --check-prefix=NOCOALESCE

#include <stdio.h>

extern "C" {

// A run of fputc to the same stream is the classic char-at-a-time antipattern.
// It coalesces into a single 4-byte fwrite via a stack gather buffer.

// COALESCE-LABEL: define {{.*}}@my_fputc
// NOCOALESCE-LABEL: define {{.*}}@my_fputc
void my_fputc(FILE* f) {
    // COALESCE: %fputc.gather.buf = alloca [4 x i8]
    // COALESCE: call i64 @fwrite(ptr {{.*}}fputc.gather.buf{{.*}}, i64 1, i64 4, ptr {{.*}})
    // COALESCE-NOT: call i32 @fputc

    // NOCOALESCE-COUNT-4: call i32 @fputc
    // NOCOALESCE-NOT: @fwrite
    fputc('H', f);
    fputc('i', f);
    fputc('!', f);
    fputc('\n', f);
}

}

