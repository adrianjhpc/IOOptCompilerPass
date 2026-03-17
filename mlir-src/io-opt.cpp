#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Passes.h"

#include "IODialect.h"

// Forward declare our custom pass registration functions
namespace mlir {
namespace io {
  void registerRecognizeIOPass();
  void registerIOPasses();
  void registerConvertIOToLLVMPass();
}
}


// The bridge burner: Safely unlinks ClangIR types, generates LLVM type conversions, and sweeps up redundant ClangIR bitcasts
struct RemoveIOCastPass : public mlir::PassWrapper<RemoveIOCastPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveIOCastPass)
  llvm::StringRef getArgument() const final { return "remove-io-cast"; }
  llvm::StringRef getDescription() const final { return "Removes io.cast ops and illegal bitcasts"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    mlir::OpBuilder builder(&getContext());
    std::vector<mlir::Operation*> opsToErase;

    getOperation()->walk([&](mlir::Operation *op) {
      // 1. Fix ClangIR's Opaque Pointer Bug (Delete redundant bitcasts)
      if (op->getName().getStringRef() == "llvm.bitcast") {
        if (op->getOperand(0).getType() == op->getResult(0).getType()) {
          op->getResult(0).replaceAllUsesWith(op->getOperand(0));
          opsToErase.push_back(op);
        }
      } 
      // 2. Clean up our custom IO bridges
      else if (op->getName().getStringRef() == "io.cast") {
        mlir::Value operand = op->getOperand(0);
        mlir::Value result = op->getResult(0);
        
        // Peer through ClangIR's temporary conversion cast to get the raw LLVM value
        if (auto castOp = operand.getDefiningOp()) {
            if (castOp->getName().getStringRef() == "builtin.unrealized_conversion_cast") {
                operand = castOp->getOperand(0);
            }
        }
        
        // If it's a pointer going into an integer array, emit standard LLVM ptrtoint
        if (operand.getType() != result.getType()) {
          builder.setInsertionPoint(op);
          if (mlir::isa<mlir::LLVM::LLVMPointerType>(operand.getType())) {
            operand = builder.create<mlir::LLVM::PtrToIntOp>(op->getLoc(), result.getType(), operand);
          }
        }
        
        result.replaceAllUsesWith(operand);
        opsToErase.push_back(op);
      }
    });

    // Safely delete everything we collected
    for (auto *op : opsToErase) {
      op->erase();
    }
  }
};

// The Healer: Runs ClangIR's conversion to fix the types before the verifier wakes up
struct CIRToLLVMInHousePass : public mlir::PassWrapper<CIRToLLVMInHousePass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CIRToLLVMInHousePass)
  llvm::StringRef getArgument() const final { return "cir-to-llvm-inhouse"; }
  llvm::StringRef getDescription() const final { return "Runs ClangIR to LLVM lowering in-house"; }

  void runOnOperation() override {
    mlir::PassManager pm(&getContext());
    
    // Disable the internal verifier because we are handing it our temporary bridging IR
    pm.enableVerifier(false); 
    
    // PassManager inherits from OpPassManager, so this works perfectly!
    cir::direct::populateCIRToLLVMPasses(pm, false);
    
    if (mlir::failed(pm.run(getOperation())))
      signalPassFailure();
  }

};

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
 
  mlir::registerAllPasses();
   
  registry.insert<mlir::func::FuncDialect,
                  mlir::scf::SCFDialect,
                  mlir::memref::MemRefDialect,
                  mlir::arith::ArithDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::io::IODialect,
                  cir::CIRDialect>();

  // Register our custom passes
  mlir::io::registerRecognizeIOPass();
  mlir::io::registerIOPasses();
  mlir::io::registerConvertIOToLLVMPass();
  mlir::PassRegistration<RemoveIOCastPass>();
  mlir::PassRegistration<CIRToLLVMInHousePass>();

  // Start the MLIR command-line tool
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "IO Optimiser tool\n", registry));
}
