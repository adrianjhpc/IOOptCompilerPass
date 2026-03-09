// RUN: clang++-20 -O2 -fno-inline -emit-llvm -S -c %s -o - | opt-20 -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | FileCheck-20 %s

#include <unistd.h>

// -------------------------------------------------------------------------
// TEST 1: PERFECTLY CONTIGUOUS OFFSETS
// -------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_pwrite_contiguous
__attribute__((noinline))
void test_pwrite_contiguous(int fd, const char* b1, const char* b2, const char* b3, const char* b4, off_t base_offset) {

    // VERIFICATION:
    // We provide 4 writes to satisfy the N>=4 profitability threshold.
    // The offsets are perfectly contiguous mathematically.
    // SCEV should prove this and emit a single 4-element pwritev.

    // CHECK: call {{.*}} @pwritev(i32 {{.*}}, ptr {{.*}}, i32 4, i64 {{.*}})
    
    // CRITICAL FIX: Add the open parenthesis so we don't accidentally match "pwritev"
    // CHECK-NOT: call {{.*}} @pwrite(

    pwrite(fd, b1, 10, base_offset);
    pwrite(fd, b2, 20, base_offset + 10);
    pwrite(fd, b3, 30, base_offset + 30);
    pwrite(fd, b4, 40, base_offset + 60);
}

// -------------------------------------------------------------------------
// TEST 2: OFFSET GAP HAZARD
// -------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_pwrite_gap
__attribute__((noinline))
void test_pwrite_gap(int fd, const char* b1, const char* b2, const char* b3, const char* b4, off_t base_offset) {

    // VERIFICATION:
    // The second write starts at `base_offset + 15` (a 5-byte gap).
    // The pass MUST NOT merge these.

    // CHECK-NOT: call {{.*}} @pwritev

    // We expect 4 separate pwrite calls because of the gaps. 
    // We use wildcards to ignore 'noundef' and register names.
    
    // CHECK: call {{.*}} @pwrite(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}10, i64 {{.*}})
    // CHECK: call {{.*}} @pwrite(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}20, i64 {{.*}})
    // CHECK: call {{.*}} @pwrite(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}30, i64 {{.*}})
    // CHECK: call {{.*}} @pwrite(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}40, i64 {{.*}})

    pwrite(fd, b1, 10, base_offset);
    pwrite(fd, b2, 20, base_offset + 15); // THE GAP HAZARD
    pwrite(fd, b3, 30, base_offset + 35);
    pwrite(fd, b4, 40, base_offset + 65);
}
