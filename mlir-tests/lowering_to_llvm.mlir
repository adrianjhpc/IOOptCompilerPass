// RUN: io-opt %s -convert-io-to-llvm | FileCheck %s

// When lowered to LLVM IR, we expect to see an external C function declaration for `write`
// at the module level, before the function definitions.
// CHECK: llvm.func @write(i32, !llvm.ptr, i64) -> i64

// CHECK-LABEL: func.func @test_lowering_batch_write
func.func @test_lowering_batch_write(%fd: index, %buf: memref<100xi8>, %size: index) {

  // Here is our optimized MLIR operation
  %bytes_written = io.batch_write %fd, %buf, %size : memref<100xi8>

  return
}

// And we expect our `io.batch_write` to be replaced by an LLVM call to that C function!
// CHECK: llvm.call @write
// CHECK-NOT: io.batch_write
