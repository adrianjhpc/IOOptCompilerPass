#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace io {

std::unique_ptr<mlir::Pass> createIOLoopBatchingPass();

void registerIOPasses();

std::unique_ptr<mlir::Pass> createConvertIOToLLVMPass();

void registerConvertIOToLLVMPass();

} // namespace io
} // namespace mlir
