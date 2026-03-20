// RUN: io-opt %s --io-loop-batching | %FileCheck %s

// An unknown external function that could do anything
func.func private @opaque_side_effect_function()

// ============================================================================
// TEST 1: The Side-Effect Bailout
// ============================================================================
// CHECK-LABEL: func.func @test_hazard_loop
func.func @test_hazard_loop(%fd: i32, %base_ptr: !llvm.ptr) {
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %step = arith.constant 1 : index
    %write_size = arith.constant 1 : i64

    // Ensure the compiler DOES NOT optimize this loop
    // CHECK: scf.for
    // CHECK: io.write
    // CHECK: func.call @opaque_side_effect_function

    scf.for %iv = %c0 to %c100 step %step {
        %iv_i64 = arith.index_cast %iv : index to i64
        %ptr = llvm.getelementptr %base_ptr[%iv_i64] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        
        // This is perfectly contiguous I/O...
        %res = io.write %fd, %ptr, %write_size : !llvm.ptr
        
        // ...but this opaque function ruins it! The compiler must abort.
        func.call @opaque_side_effect_function() : () -> ()
    }

    return
}
