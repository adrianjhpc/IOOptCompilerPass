#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

#include "IODialect.h"
#include "TargetUtils.h"

using namespace mlir;

namespace {

// Pattern to lift `write(fd, buf, count)`
struct LiftWritePattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp, PatternRewriter &rewriter) const override {
    // Check if the function being called is named "write"
    auto calleeAttr = callOp.getCalleeAttr();
    if (!calleeAttr) return failure();
    if (calleeAttr.getValue() != "write") return failure();

    // Ensure it matches the POSIX signature: ssize_t write(int fd, const void *buf, size_t count);
    if (callOp.getNumOperands() != 3) return failure();
    if (callOp.getNumResults() != 1) return failure(); // Expecting a return value (bytes written)

    Value fd = callOp.getOperand(0);
    Value buf = callOp.getOperand(1);
    Value count = callOp.getOperand(2);

    // Upgrade to our custom dialect
    auto ioWrite = io::WriteOp::create(
        rewriter,
        callOp.getLoc(),
        callOp.getResultTypes().front(), // Maintain the original return type (usually i64)
        fd,
        buf,
        count
    );


    // Replace the generic call with our semantic operation
    rewriter.replaceOp(callOp, ioWrite.getResult());
    return success();
  }
};

// Pattern to lift `read(fd, buf, count)`
struct LiftReadPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp, PatternRewriter &rewriter) const override {
    auto calleeAttr = callOp.getCalleeAttr();
    if (!calleeAttr) return failure();
    if (calleeAttr.getValue() != "read") return failure();

    if (callOp.getNumOperands() != 3) return failure();
    if (callOp.getNumResults() != 1) return failure(); 

    Value fd = callOp.getOperand(0);
    Value buf = callOp.getOperand(1);
    Value count = callOp.getOperand(2);

    auto ioRead = io::ReadOp::create(
        rewriter,
        callOp.getLoc(),
        callOp.getResultTypes().front(), 
        fd,
        buf,
        count
    );

    rewriter.replaceOp(callOp, ioRead.getResult());
    return success();
  }
};

struct RecogniseIOPass : public PassWrapper<RecogniseIOPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RecogniseIOPass)

  llvm::StringRef getArgument() const final { return "recognise-io"; }
  llvm::StringRef getDescription() const final { return "Lifts standard C library I/O calls into the custom IO dialect."; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<io::IODialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    mlir::io::bootstrapTargetInfo(module);

    RewritePatternSet patterns(context);
    patterns.add<LiftWritePattern>(context);
    patterns.add<LiftReadPattern>(context);
    
    // Apply the lifting patterns module-wide
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

// Expose it to the pass manager
namespace mlir {
namespace io {
  std::unique_ptr<mlir::Pass> createRecogniseIOPass() {
    return std::make_unique<RecogniseIOPass>();
  }
  
  void registerRecogniseIOPass() {
    PassRegistration<RecogniseIOPass>();
  }
} // namespace io
} // namespace mlir
