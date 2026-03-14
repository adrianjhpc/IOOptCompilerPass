// mlir-src/io-opt.cpp
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Include the specific core MLIR dialects we interact with
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "IODialect.h"
#include "IOPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  // Register the standard dialects we need
  registry.insert<mlir::func::FuncDialect,
                  mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect>();
  
  // Register our custom IO dialect
  registry.insert<mlir::io::IODialect>();

  // Register our custom passes so they appear on the command line
  mlir::io::registerIOPasses();
  mlir::io::registerConvertIOToLLVMPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "IO Optimiser Driver\n", registry));
}
