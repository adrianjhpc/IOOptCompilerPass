// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

// --- MOCK C TYPES ---
typedef struct _IO_FILE FILE;
extern "C" unsigned long fwrite(const void *ptr, unsigned long size, unsigned long count, FILE *stream);
// --------------------

// CHECK-LABEL: define dso_local void @_Z20test_fwrite_math_fixP8_IO_FILEPcS1_
void test_fwrite_math_fix(FILE *f, char *buf1, char *buf2) {
    // We are writing 4 items of 3 bytes = 12 bytes total
    // Followed by 2 items of 5 bytes = 10 bytes total
    // Total footprint = 22 bytes.

    // 1. Verify the pass calculates the true byte footprint for the stack allocation.
    // CHECK: %shadow.buf = alloca [22 x i8]
    
    // 2. Verify the memcpy operations grab the correct total byte lengths (12 and 10).
    // CHECK: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i64 12, i1 false)
    // CHECK: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i64 10, i1 false)
    
    // 3. Verify the final emitted fwrite normalizes size to 1, and count to 22.
    // CHECK: call i64 @fwrite(ptr {{.*}}, i64 1, i64 22, ptr {{.*}})
    
    fwrite(buf1, 3, 4, f); 
    fwrite(buf2, 5, 2, f); 
}
