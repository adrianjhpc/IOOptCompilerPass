// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s --check-prefix=COALESCE
// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -enable-io-opt=false -S | %FileCheck %s --check-prefix=NOCOALESCE

#include <stdio.h>
extern "C" {
// COALESCE-LABEL: define {{.*}}@emit4
// NOCOALESCE-LABEL: define {{.*}}@emit4
void emit4(const char* a, const char* b, const char* c, const char* d) {
    // COALESCE: call i64 @strlen
    // The appended '\n' (ASCII 10) stored into the gather buffer:
    // COALESCE: store i8 10, ptr
    // COALESCE: load ptr, ptr @stdout
    // COALESCE: call i64 @fwrite(ptr {{.*}}, i64 1, i64 {{.*}}, ptr {{.*}})
    // COALESCE-NOT: call {{.*}}@puts

    // NOCOALESCE-COUNT-4: call {{.*}}@puts
    // NOCOALESCE-NOT: @fwrite
    puts(a); puts(b); puts(c); puts(d);
}
}

