#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h" // NEEDED for MemRef to LLVM conversion
#include "mlir/Transforms/DialectConversion.h"        // NEEDED for applyPartialConversion
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "IODialect.h"

using namespace mlir;

namespace {

// Your exact lowering pattern!
struct BatchWriteLowering : public ConvertOpToLLVMPattern<io::BatchWriteOp> {
  using ConvertOpToLLVMPattern<io::BatchWriteOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(io::BatchWriteOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<ModuleOp>();

    // Ensure standard POSIX `write(int fd, void* buf, size_t count)` exists
    auto writeFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("write");
    if (!writeFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto writeType = LLVM::LLVMFunctionType::get(
          rewriter.getI64Type(),
          {rewriter.getI32Type(), LLVM::LLVMPointerType::get(rewriter.getContext()), rewriter.getI64Type()}
      );
      writeFunc = rewriter.create<LLVM::LLVMFuncOp>(op.getLoc(), "write", writeType);
    }

    // Cast the MLIR types down to raw LLVM types
    Value fdI32 = rewriter.create<LLVM::TruncOp>(op.getLoc(), rewriter.getI32Type(), adaptor.getFd());

    // Extract the raw pointer from the MLIR MemRef descriptor
    auto memrefType = mlir::cast<MemRefType>(op.getBuffer().getType());
    Value rawPtr = getStridedElementPtr(op.getLoc(), memrefType, adaptor.getBuffer(), {}, rewriter);

    // Emit the actual LLVM IR Call instruction
    auto llvmCall = rewriter.create<LLVM::CallOp>(
        op.getLoc(),
        writeFunc,
        ValueRange{fdI32, rawPtr, adaptor.getTotalSize()}
    );

    rewriter.replaceOp(op, llvmCall.getResult());
    return success();
  }
};

// NEW: The Pass that runs the pattern
struct ConvertIOToLLVMPass : public PassWrapper<ConvertIOToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertIOToLLVMPass)

  llvm::StringRef getArgument() const final { return "convert-io-to-llvm"; }
  llvm::StringRef getDescription() const final { return "Lowers the IO dialect to LLVM IR"; }

  // Tell MLIR that this pass generates LLVM dialect operations
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // The LLVMTypeConverter handles translating complex MLIR types (like MemRefs)
    // into standard LLVM structs and pointers automatically.
    LLVMTypeConverter typeConverter(context);

    // Set up the Conversion Target
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();   // LLVM output is legal
    target.addIllegalOp<io::BatchWriteOp>();       // io.batch_write must be eliminated!

    RewritePatternSet patterns(context);
    patterns.add<BatchWriteLowering>(typeConverter);

    // Apply the conversion to the module
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end anonymous namespace

// Expose the pass to the outside world
namespace mlir {
namespace io {
  std::unique_ptr<mlir::Pass> createConvertIOToLLVMPass() {
    return std::make_unique<ConvertIOToLLVMPass>();
  }
  
  void registerConvertIOToLLVMPass() {
    PassRegistration<ConvertIOToLLVMPass>();
  }
} // namespace io
} // namespace mlir
