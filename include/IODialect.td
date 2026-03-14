//===- IODialect.td - MLIR I/O Dialect Definition ----------*- tablegen -*-===//

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

// Basic I/O Operations (Matches `operation write(...)`)
def IO_ReadOp : IO_Op<"read", [MemoryEffects<[MemWrite]>]> {
  let summary = "Basic synchronous read operation";
  let arguments = (ins Index:$fd, AnyMemRef:$buffer, Index:$size);
  let results = (outs I64:$bytes_read);
  let assemblyFormat = "$fd `,` $buffer `,` $size attr-dict `:` type($buffer)";
}

def IO_WriteOp : IO_Op<"write", [MemoryEffects<[MemRead]>]> {
  let summary = "Basic synchronous write operation";
  let arguments = (ins Index:$fd, AnyMemRef:$buffer, Index:$size);
  let results = (outs I64:$bytes_written);
  let assemblyFormat = "$fd `,` $buffer `,` $size attr-dict `:` type($buffer)";
}

// Batched Operations (Matches `operation batch_write(...)`)
def IO_BatchReadOp : IO_Op<"batch_read", [MemoryEffects<[MemWrite]>]> {
  let summary = "Optimized batched read operation";
  let arguments = (ins Index:$fd, AnyMemRef:$buffer, Index:$total_size);
  let results = (outs I64:$bytes_read);
  let assemblyFormat = "$fd `,` $buffer `,` $total_size attr-dict `:` type($buffer)";
}

def IO_BatchWriteOp : IO_Op<"batch_write", [MemoryEffects<[MemRead]>]> {
  let summary = "Optimized batched write operation";
  let arguments = (ins Index:$fd, AnyMemRef:$buffer, Index:$total_size);
  let results = (outs I64:$bytes_written);
  let assemblyFormat = "$fd `,` $buffer `,` $total_size attr-dict `:` type($buffer)";
}

