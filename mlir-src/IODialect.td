include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def IODialect : Dialect {
  let name = "io";
  let summary = "A high-level dialect for I/O operation optimization.";
  let cppNamespace = "::mlir::io";
}

// Base class for all IO operations
class IO_Op<string mnemonic, list<Trait> traits = []> :
    Op<IODialect, mnemonic, traits>;

// ============================================================================
// Basic I/O Operations 
// ============================================================================
def IO_ReadOp : IO_Op<"read", [MemoryEffects<[MemWrite]>]> {
  let summary = "Basic synchronous read operation";
  // Updated: I32 for fd, I64 for size, AnyType for buffer (!llvm.ptr or memref)
  let arguments = (ins I32:$fd, AnyType:$buffer, I64:$size);
  let results = (outs I64:$bytes_read);
  let assemblyFormat = "$fd `,` $buffer `,` $size attr-dict `:` type($buffer)";
}

def IO_WriteOp : IO_Op<"write"> {
  let summary = "Basic synchronous write operation";
  let arguments = (ins I32:$fd, AnyType:$buffer, I64:$size);
  let results = (outs I64:$bytes_written);
  let assemblyFormat = "$fd `,` $buffer `,` $size attr-dict `:` type($buffer)";
}

// ============================================================================
// Contiguous Batched Operations
// ============================================================================
def IO_BatchReadOp : IO_Op<"batch_read", [MemoryEffects<[MemWrite]>]> {
  let summary = "Optimized contiguous batched read operation";
  let arguments = (ins I32:$fd, AnyType:$buffer, I64:$total_size);
  let results = (outs I64:$bytes_read);
  let assemblyFormat = "$fd `,` $buffer `,` $total_size attr-dict `:` type($buffer)";
}

def IO_BatchWriteOp : IO_Op<"batch_write"> {
  let summary = "Optimized contiguous batched write operation";
  let arguments = (ins I32:$fd, AnyType:$buffer, I64:$total_size);
  let results = (outs I64:$bytes_written);
  let assemblyFormat = "$fd `,` $buffer `,` $total_size attr-dict `:` type($buffer)";
}

// ============================================================================
// Scatter/Gather Vector Batched Operations (readv / writev)
// ============================================================================
def IO_BatchReadVOp : IO_Op<"batch_readv", [MemoryEffects<[MemWrite]>]> {
  let summary = "Optimized scatter/gather batched readv operation";
  // $ptrs and $sizes are the memrefs we generate. $count is the loop trip count (Index)
  let arguments = (ins I32:$fd, AnyType:$ptrs, AnyType:$sizes, Index:$count);
  let results = (outs I64:$bytes_read);
  let assemblyFormat = "$fd `,` $ptrs `,` $sizes `,` $count attr-dict `:` type($ptrs) `,` type($sizes)";
}

def IO_BatchWriteVOp : IO_Op<"batch_writev"> {
  let summary = "Optimized scatter/gather batched writev operation";
  let arguments = (ins I32:$fd, AnyType:$ptrs, AnyType:$sizes, Index:$count);
  let results = (outs I64:$bytes_written);
  let assemblyFormat = "$fd `,` $ptrs `,` $sizes `,` $count attr-dict `:` type($ptrs) `,` type($sizes)";
}


// ============================================================================
// Bridge IR
// ============================================================================
def IOCastOp : IO_Op<"cast"> {
  let summary = "Type bridge between ClangIR and standard MLIR dialects";
  let description = [{
    This operation acts as a legal bridge to carry ClangIR types (like !cir.int)
    into standard MLIR types (like i32) without triggering verifier crashes.
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
}

// ============================================================================
// Zero-Copy I/O Operations
// ============================================================================
def IO_SendfileOp : IO_Op<"sendfile", [MemoryEffects<[MemWrite]>]> {
  let summary = "Zero-copy data transfer between file descriptors";
  let description = [{
    Transfers data directly between two file descriptors entirely within the 
    kernel space, bypassing the need to copy data to/from user space buffers.
  }];
  
  // Arguments: out_fd, in_fd, offset (ptr), count (size)
  let arguments = (ins I32:$out_fd, I32:$in_fd, AnyType:$offset, I64:$count);
  let results = (outs I64:$bytes_written);
  
  let assemblyFormat = "$out_fd `,` $in_fd `,` $offset `,` $count attr-dict `:` type($offset)";
}

// ============================================================================
// Asynchronous I/O Operations (io_uring / aio)
// ============================================================================
def IO_SubmitOp : IO_Op<"submit"> {
  let summary = "Submits an asynchronous I/O operation";
  let description = [{
    Registers an I/O operation with the kernel to be executed in the background.
    Returns an integer 'ticket' that must be passed to an io.wait operation.
  }];
  
  // Matching your IO_ReadOp arguments
  let arguments = (ins I32:$fd, AnyType:$buffer, I64:$size);
  let results = (outs I32:$ticket);
  let assemblyFormat = "$fd `,` $buffer `,` $size attr-dict `:` type($buffer)";
}

def IO_WaitOp : IO_Op<"wait", [MemoryEffects<[MemWrite]>]> {
  let summary = "Waits for an asynchronous I/O operation to complete";
  let description = [{
    Blocks the current thread until the kernel has finished writing data to 
    the buffer associated with the given ticket.
  }];
  
  let arguments = (ins I32:$ticket);
  // Matching your IO_ReadOp results
  let results = (outs I64:$bytes_read);
  let assemblyFormat = "$ticket attr-dict";
}

// ============================================================================
// Prefetch / Read-Ahead Operations
// ============================================================================
def IO_PrefetchOp : IO_Op<"prefetch"> {
  let summary = "Asynchronous kernel read-ahead hint";
  let description = [{
    Advises the kernel to aggressively prefetch data for the given file
    descriptor. Maps down to posix_fadvise(POSIX_FADV_WILLNEED).
  }];

  // We only need the file descriptor and the lookahead size.
  let arguments = (ins I32:$fd, I64:$lookahead_size);
  
  // No results, because it's purely a background performance hint.
  let assemblyFormat = "$fd `,` $lookahead_size attr-dict";
}

// ============================================================================
// Memory Mapping Operations
// ============================================================================
def IO_MmapOp : IO_Op<"mmap"> {
  let summary = "Memory maps a file directly into user space";
  let description = [{
    Maps a file directly into the application's virtual address space. 
    This entirely bypasses user-space buffer allocation and kernel-to-user 
    data copying, providing zero-copy random access to file contents.
  }];

  // Arguments: file descriptor, length/size, and file offset
  let arguments = (ins I32:$fd, I64:$size, I64:$offset);
  
  // Returns the pointer to the newly mapped memory block
  let results = (outs AnyType:$buffer);
  
  let assemblyFormat = "$fd `,` $size `,` $offset attr-dict `:` type($buffer)";
}
