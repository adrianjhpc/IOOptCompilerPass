// RUN: io-opt %s --io-loop-batching | %FileCheck %s

!s32i = !cir.int<s, 32>
!s8i = !cir.int<s, 8>

module {
  // Mock the write function declaration
  cir.func private @write(!s32i, !cir.ptr<!s8i>, !s32i) -> !s32i

  // CHECK-LABEL: cir.func @test_step_extraction
  cir.func @test_step_extraction(%fd: !s32i, %buf: !cir.ptr<!s8i>) {
    %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
    %zero = cir.const #cir.int<0> : !s32i
    cir.store align(4) %zero, %0 : !s32i, !cir.ptr<!s32i>

    // A loop from 0 to 1000, step 2. 
    cir.for : cond {
      %i = cir.load align(4) %0 : !cir.ptr<!s32i>, !s32i
      %1000 = cir.const #cir.int<1000> : !s32i
      %cmp = cir.cmp(lt, %i, %1000) : !s32i, !cir.bool
      cir.condition(%cmp)
    } body {
      cir.scope {
        %len = cir.const #cir.int<32> : !s32i
        // The write call inside the loop
        cir.call @write(%fd, %buf, %len) : (!s32i, !cir.ptr<!s8i>, !s32i) -> !s32i
      }
      cir.yield
    } step {
      // Here is the step of 2!
      %2 = cir.const #cir.int<2> : !s32i
      %i = cir.load align(4) %0 : !cir.ptr<!s32i>, !s32i
      %next = cir.binop(add, %i, %2) : !s32i
      cir.store align(4) %next, %0 : !s32i, !cir.ptr<!s32i>
      cir.yield
    }
    
    // CHECK: %[[TRIP_COUNT:.*]] = arith.constant 500 : index
    // CHECK: io.batch_writev {{.*}}, {{.*}}, {{.*}}, %[[TRIP_COUNT]] :
    
    // CHECK-NOT: cir.call @write
    cir.return
  }
}
