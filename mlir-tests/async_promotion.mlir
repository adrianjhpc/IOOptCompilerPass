// RUN: io-opt -io-async-promotion %s | %FileCheck %s

func.func private @read(i32, !llvm.ptr, i64) -> i64
func.func private @opaque_side_effect() -> ()

// ============================================================================
// TEST 1: The Happy Path (Successful Promotion)
// ============================================================================
// CHECK-LABEL: func @test_async_promotion
func.func @test_async_promotion(%fd: i32, %buf: !llvm.ptr, %size: i64, %other_val: i64) -> i64 {
    // CHECK-NEXT: %[[TOKEN:.*]] = io.submit %arg0, %arg1, %arg2 : !llvm.ptr
    // CHECK-NOT: call @read
    %bytes_read = func.call @read(%fd, %buf, %size) : (i32, !llvm.ptr, i64) -> i64

    // CHECK-NEXT: %[[ADD:.*]] = arith.addi %arg3, %arg3
    %add = arith.addi %other_val, %other_val : i64

    // CHECK-NEXT: %[[BYTES:.*]] = io.wait %[[TOKEN]]
    // CHECK-NEXT: %[[RET:.*]] = arith.addi %[[BYTES]], %[[ADD]]
    %ret = arith.addi %bytes_read, %add : i64
    
    // CHECK-NEXT: return %[[RET]]
    func.return %ret : i64
}

// ============================================================================
// TEST 2: No Independent Compute (No Promotion)
// ============================================================================
// CHECK-LABEL: func @test_no_promotion
func.func @test_no_promotion(%fd: i32, %buf: !llvm.ptr, %size: i64) -> i64 {
    // CHECK-NEXT: %[[BYTES:.*]] = call @read
    // CHECK-NOT: io.submit
    %bytes_read = func.call @read(%fd, %buf, %size) : (i32, !llvm.ptr, i64) -> i64

    // CHECK-NEXT: %[[RET:.*]] = arith.addi %[[BYTES]], %[[BYTES]]
    %ret = arith.addi %bytes_read, %bytes_read : i64
    
    func.return %ret : i64
}

// ============================================================================
// TEST 3: The Opaque Hazard (Safe Fallback)
// ============================================================================
// CHECK-LABEL: func @test_hazard_blocks_promotion
func.func @test_hazard_blocks_promotion(%fd: i32, %buf: !llvm.ptr, %size: i64, %val: i64) -> i64 {
    // CHECK-NEXT: %[[TOKEN:.*]] = io.submit
    %bytes_read = func.call @read(%fd, %buf, %size) : (i32, !llvm.ptr, i64) -> i64

    // CHECK-NEXT: arith.addi %arg3, %arg3
    %add = arith.addi %val, %val : i64

    // HAZARD! Opaque side effect. Wait must be forced here.
    // CHECK-NEXT: %[[BYTES:.*]] = io.wait %[[TOKEN]]
    // CHECK-NEXT: call @opaque_side_effect()
    func.call @opaque_side_effect() : () -> ()

    // CHECK-NEXT: return %[[BYTES]]
    func.return %bytes_read : i64
}

// ============================================================================
// TEST 4: The Terminator Barrier
// ============================================================================
// CHECK-LABEL: func @test_terminator_barrier
func.func @test_terminator_barrier(%fd: i32, %buf: !llvm.ptr, %size: i64, %val: i64) -> i64 {
    // CHECK-NEXT: %[[TOKEN:.*]] = io.submit
    %bytes_read = func.call @read(%fd, %buf, %size) : (i32, !llvm.ptr, i64) -> i64

    // CHECK-NEXT: arith.addi %arg3, %arg3
    %add = arith.addi %val, %val : i64

    // HAZARD! Terminator. Wait must be forced here.
    // CHECK-NEXT: %[[BYTES:.*]] = io.wait %[[TOKEN]]
    // CHECK-NEXT: return %[[BYTES]]
    func.return %bytes_read : i64
}
