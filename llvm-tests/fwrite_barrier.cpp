// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <cstdio>

// CHECK-LABEL: define dso_local void @_Z19test_fwrite_barrierP8_IO_FILEPcS1_
void test_fwrite_barrier(FILE* f, char* buf1, char* buf2) {
    // The fseek instruction must trigger a flushBatch!
    // CHECK: call i64 @fwrite(ptr {{.*}}, i64 {{.*}}10, i64 {{.*}}1, ptr {{.*}})
    // CHECK: call i32 @fseek(ptr {{.*}}, i64 {{.*}}0, i32 {{.*}}0)
    // CHECK: call i64 @fwrite(ptr {{.*}}, i64 {{.*}}10, i64 {{.*}}1, ptr {{.*}})
    // CHECK-NOT: shadow.buf
    
    fwrite(buf1, 10, 1, f);
    
    // Barrier!
    fseek(f, 0, SEEK_SET); 
    
    fwrite(buf2, 10, 1, f);
}
