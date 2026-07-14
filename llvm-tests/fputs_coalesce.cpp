// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s --check-prefix=COALESCE
// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -enable-io-opt=false -S | %FileCheck %s --check-prefix=NOCOALESCE

#include <stdio.h>
extern "C" {
// COALESCE-LABEL: define {{.*}}@log4
// NOCOALESCE-LABEL: define {{.*}}@log4
void log4(FILE* f, const char* a, const char* b, const char* c, const char* d) {
    // COALESCE: call i64 @strlen
    // COALESCE: call i32 @posix_memalign
    // COALESCE: call i64 @fwrite(ptr {{.*}}, i64 1, i64 {{.*}}, ptr {{.*}})
    // COALESCE: call void @free
    // COALESCE-NOT: call {{.*}}@fputs

    // NOCOALESCE-COUNT-4: call {{.*}}@fputs
    // NOCOALESCE-NOT: @fwrite
    fputs(a, f); fputs(b, f); fputs(c, f); fputs(d, f);
}
}

