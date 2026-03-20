// RUN: io-opt %s --io-loop-batching | %FileCheck %s

!s32i = !cir.int<s, 32>
!s8i = !cir.int<s, 8>

module {
  // Mock the standard read function declaration
  cir.func private @read(!s32i, !cir.ptr<!s8i>, !s32i) -> !s32i

  // CHECK-LABEL: cir.func @test_read_batching
  cir.func @test_read_batching(%fd: !s32i, %buf: !cir.ptr<!s8i>) {
    %i_ptr = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
    %zero = cir.const #cir.int<0> : !s32i
    cir.store align(4) %zero, %i_ptr : !s32i, !cir.ptr<!s32i>

    // Loop from 0 to 100, step 1
    cir.for : cond {
      %i = cir.load align(4) %i_ptr : !cir.ptr<!s32i>, !s32i
      %limit = cir.const #cir.int<100> : !s32i
      %cmp = cir.cmp(lt, %i, %limit) : !s32i, !cir.bool
      cir.condition(%cmp)
    } body {
      cir.scope {
        %len = cir.const #cir.int<64> : !s32i
        // The target read call
        cir.call @read(%fd, %buf, %len) : (!s32i, !cir.ptr<!s8i>, !s32i) -> !s32i
      }
      cir.yield
    } step {
      %one = cir.const #cir.int<1> : !s32i
      %i = cir.load align(4) %i_ptr : !cir.ptr<!s32i>, !s32i
      %next = cir.binop(add, %i, %one) : !s32i
      cir.store align(4) %next, %i_ptr : !s32i, !cir.ptr<!s32i>
      cir.yield
    }

    // --- CHECK ASSERTIONS ---
    
    // Verify we calculated the correct trip count (100 iterations)
    // CHECK: %[[COUNT:.*]] = arith.constant 100 : index
    
    // Verify we allocated the pointer and size arrays
    // CHECK: %[[PTRS:.*]] = memref.alloca(%[[COUNT]]) : memref<?xi64>
    // CHECK: %[[SIZES:.*]] = memref.alloca(%[[COUNT]]) : memref<?xi64>

    // Verify the final operation is a BATCH READ, not a write
    // CHECK: %[[FINAL_FD:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<1xi32>
    // CHECK: io.batch_readv %[[FINAL_FD]], %[[PTRS]], %[[SIZES]], %[[COUNT]] : 
    
    // Ensure the original individual read call is gone
    // CHECK-NOT: cir.call @read
    
    cir.return
  }
}
