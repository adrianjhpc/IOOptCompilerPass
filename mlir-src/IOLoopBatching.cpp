#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "IODialect.h"

using namespace mlir;

namespace {

/// PASS 1: Loop I/O Pattern Recognition
/// Transforms loops with repeated I/O operations into batched operations.
/// 
/// Before: scf.for %i = %zero to %ten { io.write(%fd, %buffer, %one) }
/// After:  io.batch_write(%fd, %buffer, %ten)
struct HoistIOLoopPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop, PatternRewriter &rewriter) const override {
    Block *body = loop.getBody();
    if (!body || body->empty()) return failure();

    io::WriteOp writeOp = nullptr;
    bool hasSideEffects = false;

    // Detect if this loop is a pure I/O loop
    for (Operation &op : *body) {
      if (isa<scf::YieldOp>(op)) continue; 

      if (auto ioWrite = dyn_cast<io::WriteOp>(op)) {
        if (writeOp) { hasSideEffects = true; break; } // Abort if complex multi-write
        writeOp = ioWrite;
      } else if (!isMemoryEffectFree(&op)) {
        hasSideEffects = true; break; // Abort if there are unknown side-effects
      }
    }

    if (hasSideEffects || !writeOp) return failure();

    // Validate that the File Descriptor and Buffer exist outside the loop
    if (!loop.isDefinedOutsideOfLoop(writeOp.getFd()) ||
        !loop.isDefinedOutsideOfLoop(writeOp.getBuffer())) {
      return failure();
    }

    Location loc = loop.getLoc();
    
    // Mathematically calculate Trip Count: (UpperBound - LowerBound) / Step
    Value diff = rewriter.create<arith::SubIOp>(loc, loop.getUpperBound(), loop.getLowerBound());
    Value tripCount = rewriter.create<arith::DivSIOp>(loc, diff, loop.getStep());

    // Calculate Total Batch Size: TripCount * IterationSize
    Value totalSize = rewriter.create<arith::MulIOp>(loc, tripCount, writeOp.getSize());

    // Set insertion point just before the loop
    rewriter.setInsertionPoint(loop);
    
    // Create the optimized io.batch_write operation
    rewriter.create<io::BatchWriteOp>(
        loc, 
        rewriter.getI64Type(), 
        writeOp.getFd(), 
        writeOp.getBuffer(), 
        totalSize
    );

    // Completely erase the original loop
    rewriter.eraseOp(loop);

    return success();
  }
};

/// The MLIR Pass Wrapper
struct IOLoopBatchingPass : public PassWrapper<IOLoopBatchingPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IOLoopBatchingPass)

  llvm::StringRef getArgument() const final { return "io-loop-batching"; }
  llvm::StringRef getDescription() const final { return "Hoists and batches I/O operations from scf.for loops."; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<HoistIOLoopPattern>(context);
    
    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

namespace mlir {
namespace io {

// Expose the constructor
std::unique_ptr<mlir::Pass> createIOLoopBatchingPass() {
  return std::make_unique<IOLoopBatchingPass>(); 
}

// Register the pass so `io-opt` knows it exists
void registerIOPasses() {
  mlir::PassRegistration<IOLoopBatchingPass>();
}

} // namespace io
} // namespace mlir
