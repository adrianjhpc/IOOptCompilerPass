// RUN: io-opt %s --io-loop-batching | %FileCheck %s

// ============================================================================
// TEST 1: The Contiguous Read Fast-Path
// ============================================================================
// CHECK-LABEL: func.func @test_contiguous_read
func.func @test_contiguous_read(%fd: i32, %base_ptr: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %step = arith.constant 1 : index
    %read_size = arith.constant 1 : i64

    // Ensure the loop is erased
    // CHECK-NOT: scf.for

    // Ensure we emit the batched read with the constant-folded size
    // CHECK: %[[TOTAL_SIZE:.*]] = arith.constant 100 : i64
    // CHECK: io.batch_read %arg0, %arg1, %[[TOTAL_SIZE]] : !llvm.ptr

    scf.for %iv = %c0 to %c100 step %step {
        %iv_i64 = arith.index_cast %iv : index to i64
        %ptr = llvm.getelementptr %base_ptr[%iv_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %res = io.read %fd, %ptr, %read_size : !llvm.ptr
    }

    return
}

// ============================================================================
// TEST 2: The Strided Read Vector Fallback (Gather)
// ============================================================================
// CHECK-LABEL: func.func @test_strided_read
func.func @test_strided_read(%fd: i32, %base_ptr: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %step = arith.constant 2 : index   // <-- STRIDED
    %read_size = arith.constant 1 : i64

    // Ensure we allocate tracking arrays and build the Gather struct
    // CHECK: memref.alloca
    // CHECK: memref.alloca
    // CHECK: scf.for
    // CHECK: io.batch_readv %arg0, %{{.*}}, %{{.*}}, %{{.*}} : memref<?x!llvm.ptr>, memref<?xi64>

    scf.for %iv = %c0 to %c100 step %step {
        %iv_i64 = arith.index_cast %iv : index to i64
        %ptr = llvm.getelementptr %base_ptr[%iv_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %res = io.read %fd, %ptr, %read_size : !llvm.ptr
    }

    return
}
