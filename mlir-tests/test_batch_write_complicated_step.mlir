// RUN: io-opt %s --io-loop-batching | %FileCheck %s

!s32i = !cir.int<s, 32>
!s8i = !cir.int<s, 8>

module {
  cir.func private @write(!s32i, !cir.ptr<!s8i>, !s32i) -> !s32i

  // --- TEST 1: Step of 3 (1000 / 3 = 333.33 -> 334 iterations) ---
  // CHECK-LABEL: cir.func @test_step_3
  cir.func @test_step_3(%fd: !s32i, %buf: !cir.ptr<!s8i>) {
    %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
    %zero = cir.const #cir.int<0> : !s32i
    cir.store align(4) %zero, %0 : !s32i, !cir.ptr<!s32i>

    cir.for : cond {
      %i = cir.load align(4) %0 : !cir.ptr<!s32i>, !s32i
      %1000 = cir.const #cir.int<1000> : !s32i
      %cmp = cir.cmp(lt, %i, %1000) : !s32i, !cir.bool
      cir.condition(%cmp)
    } body {
      cir.scope {
        %len = cir.const #cir.int<32> : !s32i
        cir.call @write(%fd, %buf, %len) : (!s32i, !cir.ptr<!s8i>, !s32i) -> !s32i
      }
      cir.yield
    } step {
      %3 = cir.const #cir.int<3> : !s32i
      %i = cir.load align(4) %0 : !cir.ptr<!s32i>, !s32i
      %next = cir.binop(add, %i, %3) : !s32i
      cir.store align(4) %next, %0 : !s32i, !cir.ptr<!s32i>
      cir.yield
    }
    
    // CHECK: %[[TRIP_COUNT_1:.*]] = arith.constant 334 : index
    // CHECK: io.batch_writev {{.*}}, {{.*}}, {{.*}}, %[[TRIP_COUNT_1]]
    cir.return
  }

  // --- TEST 2: Step of 7 with bound of 50 (50 / 7 = 7.14 -> 8 iterations) ---
  // CHECK-LABEL: cir.func @test_step_7
  cir.func @test_step_7(%fd: !s32i, %buf: !cir.ptr<!s8i>) {
    %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
    %zero = cir.const #cir.int<0> : !s32i
    cir.store align(4) %zero, %0 : !s32i, !cir.ptr<!s32i>

    cir.for : cond {
      %i = cir.load align(4) %0 : !cir.ptr<!s32i>, !s32i
      %50 = cir.const #cir.int<50> : !s32i
      %cmp = cir.cmp(lt, %i, %50) : !s32i, !cir.bool
      cir.condition(%cmp)
    } body {
      cir.scope {
        %len = cir.const #cir.int<32> : !s32i
        cir.call @write(%fd, %buf, %len) : (!s32i, !cir.ptr<!s8i>, !s32i) -> !s32i
      }
      cir.yield
    } step {
      %7 = cir.const #cir.int<7> : !s32i
      %i = cir.load align(4) %0 : !cir.ptr<!s32i>, !s32i
      %next = cir.binop(add, %i, %7) : !s32i
      cir.store align(4) %next, %0 : !s32i, !cir.ptr<!s32i>
      cir.yield
    }
    
    // CHECK: %[[TRIP_COUNT_2:.*]] = arith.constant 8 : index
    // CHECK: io.batch_writev {{.*}}, {{.*}}, {{.*}}, %[[TRIP_COUNT_2]]
    cir.return
  }
 
  // --- TEST 3: Multiplicative Loop (MUST NOT BE OPTIMIZED) ---
  // CHECK-LABEL: cir.func @test_step_mul
  cir.func @test_step_mul(%fd: !s32i, %buf: !cir.ptr<!s8i>) {
    %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
    %zero = cir.const #cir.int<1> : !s32i
    cir.store align(4) %zero, %0 : !s32i, !cir.ptr<!s32i>

    cir.for : cond {
      %i = cir.load align(4) %0 : !cir.ptr<!s32i>, !s32i
      %1000 = cir.const #cir.int<1000> : !s32i
      %cmp = cir.cmp(lt, %i, %1000) : !s32i, !cir.bool
      cir.condition(%cmp)
    } body {
      cir.scope {
        %len = cir.const #cir.int<32> : !s32i
        cir.call @write(%fd, %buf, %len) : (!s32i, !cir.ptr<!s8i>, !s32i) -> !s32i
      }
      cir.yield
    } step {
      %2 = cir.const #cir.int<2> : !s32i
      %i = cir.load align(4) %0 : !cir.ptr<!s32i>, !s32i
      // NOTE THE MULTIPLICATION HERE!
      %next = cir.binop(mul, %i, %2) : !s32i 
      cir.store align(4) %next, %0 : !s32i, !cir.ptr<!s32i>
      cir.yield
    }
    
    // FileCheck ensures the original write call is STILL THERE, untouched!
    // CHECK: cir.call @write
    // CHECK-NOT: io.batch_writev
    cir.return
  }

}
