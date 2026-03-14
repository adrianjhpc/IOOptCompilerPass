// RUN: io-opt %s -io-loop-batching | FileCheck %s

// CHECK-LABEL: func.func @test_simple_loop_batching
func.func @test_simple_loop_batching(%fd: index, %buf: memref<100xi8>, %size: index) {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %c10 step %c1 {
    io.write %fd, %buf, %size : memref<100xi8>
  }
  return
}

// CHECK: io.batch_write
// CHECK-NOT: scf.for
