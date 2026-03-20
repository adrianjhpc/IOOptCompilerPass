// RUN: io-opt -io-prefetch-injection %s | %FileCheck %s

// Declare the mock read functions
func.func private @read(i32, !llvm.ptr, i32) -> i64
func.func private @read64(i32, !llvm.ptr, i64) -> i64

// Declare mock consumer functions to prevent Dead Code Elimination!
func.func private @consume32(i32)
func.func private @consume64(i64)

// ============================================================================
// TEST 1: The Happy Path (Successful Injection with Type Extension)
// ============================================================================
// CHECK-LABEL: func @test_prefetch_injection
func.func @test_prefetch_injection(%fd: i32, %buf: !llvm.ptr, %size: i32, %ub: index) {
    // CHECK: %[[C4:.*]] = arith.constant 4 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    // CHECK: scf.for
    scf.for %iv = %c0 to %ub step %c1 {
        // CHECK: %[[EXT:.*]] = arith.extui %arg2 : i32 to i64
        // CHECK-NEXT: %[[LOOKAHEAD:.*]] = arith.muli %[[EXT]], %[[C4]] : i64
        // CHECK-NEXT: io.prefetch %arg0, %[[LOOKAHEAD]]
        // CHECK-NEXT: call @read
        %bytes = func.call @read(%fd, %buf, %size) : (i32, !llvm.ptr, i32) -> i64

        // 10 dummy compute operations to trigger the profitability check
        %0 = arith.addi %size, %size : i32
        %1 = arith.addi %0, %size : i32
        %2 = arith.addi %1, %size : i32
        %3 = arith.addi %2, %size : i32
        %4 = arith.addi %3, %size : i32
        %5 = arith.addi %4, %size : i32
        %6 = arith.addi %5, %size : i32
        %7 = arith.addi %6, %size : i32
        %8 = arith.addi %7, %size : i32
        %9 = arith.addi %8, %size : i32
        
        // Pass the result to an opaque call so DCE doesn't delete our math!
        func.call @consume32(%9) : (i32) -> ()
    }
    func.return
}

// ============================================================================
// TEST 2: Profitability Check (Not Enough Compute)
// ============================================================================
// CHECK-LABEL: func @test_no_prefetch_short_loop
func.func @test_no_prefetch_short_loop(%fd: i32, %buf: !llvm.ptr, %size: i32, %ub: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    scf.for %iv = %c0 to %ub step %c1 {
        // CHECK-NOT: io.prefetch
        // CHECK: call @read
        %bytes = func.call @read(%fd, %buf, %size) : (i32, !llvm.ptr, i32) -> i64
        
        // Only 2 compute ops (Fails the >= 10 check)
        %0 = arith.addi %size, %size : i32
        %1 = arith.addi %0, %size : i32
        func.call @consume32(%1) : (i32) -> ()
    }
    func.return
}

// ============================================================================
// TEST 3: Native i64 Size (No Cast Needed)
// ============================================================================
// CHECK-LABEL: func @test_prefetch_i64_size
func.func @test_prefetch_i64_size(%fd: i32, %buf: !llvm.ptr, %size: i64, %ub: index) {
    // CHECK: %[[C4:.*]] = arith.constant 4 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    // CHECK: scf.for
    scf.for %iv = %c0 to %ub step %c1 {
        // CHECK-NOT: arith.extui
        // CHECK: %[[LOOKAHEAD:.*]] = arith.muli %arg2, %[[C4]] : i64
        // CHECK-NEXT: io.prefetch %arg0, %[[LOOKAHEAD]]
        // CHECK-NEXT: call @read64
        %bytes = func.call @read64(%fd, %buf, %size) : (i32, !llvm.ptr, i64) -> i64

        // 10 ops to pass profitability
        %0 = arith.addi %size, %size : i64
        %1 = arith.addi %0, %size : i64
        %2 = arith.addi %1, %size : i64
        %3 = arith.addi %2, %size : i64
        %4 = arith.addi %3, %size : i64
        %5 = arith.addi %4, %size : i64
        %6 = arith.addi %5, %size : i64
        %7 = arith.addi %6, %size : i64
        %8 = arith.addi %7, %size : i64
        %9 = arith.addi %8, %size : i64
        
        // Block DCE!
        func.call @consume64(%9) : (i64) -> ()
    }
    func.return
}
