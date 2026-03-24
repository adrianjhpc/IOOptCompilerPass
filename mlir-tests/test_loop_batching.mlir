// RUN: io-opt %s --io-loop-batching | %FileCheck %s

// ============================================================================
// TEST 1: The Contiguous Fast-Path
// ============================================================================
// CHECK-LABEL: func.func @test_contiguous_write
func.func @test_contiguous_write(%fd: i32, %base_ptr: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %step = arith.constant 1 : index
    %write_size = arith.constant 1 : i64

    // Ensure the original scf.for loop is completely erased
    // CHECK-NOT: scf.for

    // Ensure the compiler constant-folded the trip count and size to 100
    // CHECK: %[[TOTAL_SIZE:.*]] = arith.muli

    // Ensure we emit the massive batched write using the base pointer
    // CHECK: io.batch_write {{.*}}, {{.*}}, %[[TOTAL_SIZE]] 

    scf.for %iv = %c0 to %c100 step %step {
        // Cast the loop index to a concrete 64-bit integer for LLVM
        %iv_i64 = arith.index_cast %iv : index to i64
        
        // Use the i64 value for the pointer offset
        %ptr = llvm.getelementptr %base_ptr[%iv_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %res = io.write %fd, %ptr, %write_size : !llvm.ptr
    }

    return
}

// ============================================================================
// TEST 2: The Strided Vector Fallback
// ============================================================================
// CHECK-LABEL: func.func @test_strided_write
func.func @test_strided_write(%fd: i32, %base_ptr: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %step = arith.constant 2 : index   // <-- WARNING: STEP IS 2
    %write_size = arith.constant 1 : i64

    // Ensure we allocate the tracking arrays
    // CHECK: memref.alloca
    // CHECK: memref.alloca

    // Ensure we generate a new loop just to calculate the addresses
    // CHECK: scf.for
    // CHECK: llvm.getelementptr
    // CHECK: memref.store
    // CHECK: memref.store

    // Ensure the io.write inside the loop is gone...
    // CHECK-NOT: io.write

    // ...and replaced with our Scatter/Gather I/O operation outside the loop
    // CHECK: io.batch_writev %arg0, %{{.*}}, %{{.*}}, %{{.*}} : memref<?x!llvm.ptr>, memref<?xi64>

    scf.for %iv = %c0 to %c100 step %step {
        %iv_i64 = arith.index_cast %iv : index to i64
        %ptr = llvm.getelementptr %base_ptr[%iv_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %res = io.write %fd, %ptr, %write_size : !llvm.ptr
    }

    return
} 
