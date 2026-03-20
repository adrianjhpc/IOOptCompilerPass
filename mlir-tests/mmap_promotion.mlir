// RUN: io-opt -io-mmap-promotion %s | %FileCheck %s

// Mock standard C library and read functions
func.func private @malloc(i64) -> !llvm.ptr
func.func private @malloc32(i32) -> !llvm.ptr
func.func private @read(i32, !llvm.ptr, i64) -> i64
func.func private @read32(i32, !llvm.ptr, i32) -> i64
func.func private @dummy_alloc() -> !llvm.ptr

// ============================================================================
// TEST 1: The Happy Path (Perfect i64 Match)
// ============================================================================
// Expectation: Both the malloc and read are deleted. An io.mmap is created 
// with offset 0. The function returns the requested size.
// CHECK-LABEL: func @test_mmap_happy_path
func.func @test_mmap_happy_path(%fd: i32, %size: i64) -> i64 {
    // CHECK-NOT: call @malloc
    // CHECK: %[[C0:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[MMAP_PTR:.*]] = io.mmap %arg0, %arg1, %[[C0]] : !llvm.ptr
    // CHECK-NOT: call @read
    
    %ptr = func.call @malloc(%size) : (i64) -> !llvm.ptr
    %bytes_read = func.call @read(%fd, %ptr, %size) : (i32, !llvm.ptr, i64) -> i64
    
    // CHECK: return %arg1 : i64
    func.return %bytes_read : i64
}

// ============================================================================
// TEST 2: Type Extension (i32 to i64 bounds safety)
// ============================================================================
// Expectation: The size is i32, but io.mmap strictly requires i64. The pass 
// must inject an extui cast for the mmap size, AND another one for the return value.
// CHECK-LABEL: func @test_mmap_type_extension
func.func @test_mmap_type_extension(%fd: i32, %size: i32) -> i64 {
    // CHECK: %[[C0:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[SIZE64:.*]] = arith.extui %arg1 : i32 to i64
    // CHECK-NEXT: %[[MMAP_PTR:.*]] = io.mmap %arg0, %[[SIZE64]], %[[C0]] : !llvm.ptr
    
    %ptr = func.call @malloc32(%size) : (i32) -> !llvm.ptr
    %bytes_read = func.call @read32(%fd, %ptr, %size) : (i32, !llvm.ptr, i32) -> i64
    
    // CHECK-NEXT: %[[RET64:.*]] = arith.extui %arg1 : i32 to i64
    // CHECK-NEXT: return %[[RET64]] : i64
    func.return %bytes_read : i64
}

// ============================================================================
// TEST 3: Negative Test (Size Mismatch blocks promotion)
// ============================================================================
// Expectation: The user allocates %alloc_size, but reads %read_size. Since 
// this is a partial read, mmap is unsafe. The compiler must leave the code alone.
// CHECK-LABEL: func @test_mmap_size_mismatch
func.func @test_mmap_size_mismatch(%fd: i32, %alloc_size: i64, %read_size: i64) -> i64 {
    // CHECK: call @malloc
    // CHECK: call @read
    // CHECK-NOT: io.mmap
    
    %ptr = func.call @malloc(%alloc_size) : (i64) -> !llvm.ptr
    %bytes_read = func.call @read(%fd, %ptr, %read_size) : (i32, !llvm.ptr, i64) -> i64
    
    func.return %bytes_read : i64
}

// ============================================================================
// TEST 4: Negative Test (Pointer Trace Failure blocks promotion)
// ============================================================================
// Expectation: The buffer passed to read didn't originate from a malloc 
// call of the same size. The compiler must leave the code alone.
// CHECK-LABEL: func @test_mmap_pointer_mismatch
func.func @test_mmap_pointer_mismatch(%fd: i32, %size: i64) -> i64 {
    // CHECK: call @malloc
    // CHECK: call @read
    // CHECK-NOT: io.mmap
    
    // Allocate the buffer we track
    %tracked_ptr = func.call @malloc(%size) : (i64) -> !llvm.ptr
    
    // Some other buffer appears
    %untracked_ptr = func.call @dummy_alloc() : () -> !llvm.ptr
    
    // Read into the untracked buffer
    %bytes_read = func.call @read(%fd, %untracked_ptr, %size) : (i32, !llvm.ptr, i64) -> i64
    
    func.return %bytes_read : i64
}
