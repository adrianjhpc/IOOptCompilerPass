// RUN: io-opt %s --recognise-io | %FileCheck %s

// 1. Declare the external POSIX C functions (this is what ClangIR/Polygeist emits)
func.func private @write(i32, !llvm.ptr, i64) -> i64
func.func private @read(i32, !llvm.ptr, i64) -> i64

// CHECK-LABEL: func.func @test_lifting
func.func @test_lifting(%fd: i32, %buf: !llvm.ptr, %size: i64) {
    
    // Ensure the raw func.call is completely gone...
    // CHECK-NOT: func.call @write
    // ...and replaced with our semantic IO dialect
    // CHECK: %{{.*}} = io.write(%arg0, %arg1, %arg2) : i32, !llvm.ptr, i64 -> i64
    %0 = func.call @write(%fd, %buf, %size) : (i32, !llvm.ptr, i64) -> i64

    // CHECK-NOT: func.call @read
    // CHECK: %{{.*}} = io.read(%arg0, %arg1, %arg2) : i32, !llvm.ptr, i64 -> i64
    %1 = func.call @read(%fd, %buf, %size) : (i32, !llvm.ptr, i64) -> i64

    return
}
