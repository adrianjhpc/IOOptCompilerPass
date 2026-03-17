#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace io {

// Stamps the ModuleOp with the target triple and data layout for LLVM lowering.
void bootstrapTargetInfo(ModuleOp module);

} // namespace io
} // namespace mlir
