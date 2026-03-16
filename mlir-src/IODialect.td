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
