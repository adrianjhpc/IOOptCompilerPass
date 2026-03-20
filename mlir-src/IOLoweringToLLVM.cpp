#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h" // NEEDED for MemRef to LLVM conversion
#include "mlir/Transforms/DialectConversion.h"        // NEEDED for applyPartialConversion
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"

#include "IODialect.h"
#include "TargetUtils.h"

using namespace mlir;

namespace {

// ============================================================================
// Helper: Get or Insert C-Library Function Declarations
// ============================================================================
static LLVM::LLVMFuncOp getOrInsertFunc(ConversionPatternRewriter &rewriter,
                                        ModuleOp module, StringRef name,
                                        LLVM::LLVMFunctionType type) {
    if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
        return func;

    // If it doesn't exist, insert it at the top of the module
    OpBuilder::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    return rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
}

// ============================================================================
// Existing Lowerings (BatchRead/Write)
// ============================================================================

struct BatchWriteLowering : public ConvertOpToLLVMPattern<io::BatchWriteOp> {
  using ConvertOpToLLVMPattern<io::BatchWriteOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(io::BatchWriteOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<ModuleOp>();

    auto writeType = LLVM::LLVMFunctionType::get(
        rewriter.getI64Type(),
        {rewriter.getI32Type(), LLVM::LLVMPointerType::get(rewriter.getContext()), rewriter.getI64Type()}
    );
    auto writeFunc = getOrInsertFunc(rewriter, module, "write", writeType);

    Value fdI32 = rewriter.create<LLVM::TruncOp>(op.getLoc(), rewriter.getI32Type(), adaptor.getFd());

    Value rawPtr;
    if (auto memrefType = mlir::dyn_cast<MemRefType>(op.getBuffer().getType())) {
        rawPtr = getStridedElementPtr(rewriter, op.getLoc(), memrefType, adaptor.getBuffer(), {});
    } else {
        rawPtr = adaptor.getBuffer();
    }

    auto llvmCall = rewriter.create<LLVM::CallOp>(
        op.getLoc(), writeFunc, ValueRange{fdI32, rawPtr, adaptor.getTotalSize()}
    );

    rewriter.replaceOp(op, llvmCall.getResult());
    return success();
  }
};

struct BatchWriteVLowering : public ConvertOpToLLVMPattern<io::BatchWriteVOp> {
  using ConvertOpToLLVMPattern<io::BatchWriteVOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(io::BatchWriteVOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();

    Type voidPtrTy = LLVM::LLVMPointerType::get(ctx);
    Type sizeTy = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();
    Type iovecTy = LLVM::LLVMStructType::getLiteral(ctx, {voidPtrTy, sizeTy});

    auto writevType = LLVM::LLVMFunctionType::get(sizeTy, {i32Ty, voidPtrTy, i32Ty});
    auto writevFunc = getOrInsertFunc(rewriter, module, "writev", writevType);

    Value fdI32 = adaptor.getFd();
    Value vectorCountI32 = rewriter.create<LLVM::TruncOp>(loc, i32Ty, adaptor.getCount());

    Value iovecArrayPtr = rewriter.create<LLVM::AllocaOp>(
        loc, voidPtrTy, iovecTy, vectorCountI32, /*alignment=*/8);

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value countIndex = op.getCount(); 

    auto loop = rewriter.create<scf::ForOp>(loc, zero, countIndex, one);
    
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value ivI64 = rewriter.create<arith::IndexCastOp>(loc, sizeTy, iv);

    Value ptrValI64 = rewriter.create<memref::LoadOp>(loc, op.getPtrs(), ValueRange{iv});
    Value ptrVal = rewriter.create<LLVM::IntToPtrOp>(loc, voidPtrTy, ptrValI64);
    Value sizeVal = rewriter.create<memref::LoadOp>(loc, op.getSizes(), ValueRange{iv});

    Value iovAddr = rewriter.create<LLVM::GEPOp>(loc, voidPtrTy, iovecTy, iovecArrayPtr, ivI64);

    Value iovBaseAddr = rewriter.create<LLVM::GEPOp>(
        loc, voidPtrTy, iovecTy, iovAddr, ArrayRef<LLVM::GEPArg>{0, 0});
    rewriter.create<LLVM::StoreOp>(loc, ptrVal, iovBaseAddr);

    Value iovLenAddr = rewriter.create<LLVM::GEPOp>(
        loc, voidPtrTy, iovecTy, iovAddr, ArrayRef<LLVM::GEPArg>{0, 1});
    rewriter.create<LLVM::StoreOp>(loc, sizeVal, iovLenAddr);

    rewriter.setInsertionPointAfter(loop);

    auto llvmCall = rewriter.create<LLVM::CallOp>(
        loc, writevFunc, ValueRange{fdI32, iovecArrayPtr, vectorCountI32});

    rewriter.replaceOp(op, llvmCall.getResult());
    return success();
  }
};

struct BatchReadLowering : public ConvertOpToLLVMPattern<io::BatchReadOp> {
  using ConvertOpToLLVMPattern<io::BatchReadOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(io::BatchReadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();

    auto readType = LLVM::LLVMFunctionType::get(
        rewriter.getI64Type(),
        {rewriter.getI32Type(), LLVM::LLVMPointerType::get(rewriter.getContext()), rewriter.getI64Type()}
    );
    auto readFunc = getOrInsertFunc(rewriter, module, "read", readType);

    Value fdI32 = rewriter.create<LLVM::TruncOp>(op.getLoc(), rewriter.getI32Type(), adaptor.getFd());
    
    Value rawPtr;
    if (auto memrefType = mlir::dyn_cast<MemRefType>(op.getBuffer().getType())) {
        rawPtr = getStridedElementPtr(rewriter, op.getLoc(), memrefType, adaptor.getBuffer(), {});
    } else {
        rawPtr = adaptor.getBuffer();
    }

    auto llvmCall = rewriter.create<LLVM::CallOp>(
        op.getLoc(), readFunc, ValueRange{fdI32, rawPtr, adaptor.getTotalSize()}
    );

    rewriter.replaceOp(op, llvmCall.getResult());
    return success();
  }
};

struct BatchReadVLowering : public ConvertOpToLLVMPattern<io::BatchReadVOp> {
  using ConvertOpToLLVMPattern<io::BatchReadVOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(io::BatchReadVOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();

    Type voidPtrTy = LLVM::LLVMPointerType::get(ctx);
    Type sizeTy = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();
    Type iovecTy = LLVM::LLVMStructType::getLiteral(ctx, {voidPtrTy, sizeTy});

    auto readvType = LLVM::LLVMFunctionType::get(sizeTy, {i32Ty, voidPtrTy, i32Ty});
    auto readvFunc = getOrInsertFunc(rewriter, module, "readv", readvType);

    Value fdI32 = adaptor.getFd();
    Value vectorCountI32 = rewriter.create<LLVM::TruncOp>(loc, i32Ty, adaptor.getCount());

    Value iovecArrayPtr = rewriter.create<LLVM::AllocaOp>(
        loc, voidPtrTy, iovecTy, vectorCountI32, /*alignment=*/8);

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value countIndex = op.getCount(); 

    auto loop = rewriter.create<scf::ForOp>(loc, zero, countIndex, one);
    
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value ivI64 = rewriter.create<arith::IndexCastOp>(loc, sizeTy, iv);

    Value ptrValI64 = rewriter.create<memref::LoadOp>(loc, op.getPtrs(), ValueRange{iv});
    Value ptrVal = rewriter.create<LLVM::IntToPtrOp>(loc, voidPtrTy, ptrValI64);
    Value sizeVal = rewriter.create<memref::LoadOp>(loc, op.getSizes(), ValueRange{iv});

    Value iovAddr = rewriter.create<LLVM::GEPOp>(loc, voidPtrTy, iovecTy, iovecArrayPtr, ivI64);

    Value iovBaseAddr = rewriter.create<LLVM::GEPOp>(
        loc, voidPtrTy, iovecTy, iovAddr, ArrayRef<LLVM::GEPArg>{0, 0});
    rewriter.create<LLVM::StoreOp>(loc, ptrVal, iovBaseAddr);

    Value iovLenAddr = rewriter.create<LLVM::GEPOp>(
        loc, voidPtrTy, iovecTy, iovAddr, ArrayRef<LLVM::GEPArg>{0, 1});
    rewriter.create<LLVM::StoreOp>(loc, sizeVal, iovLenAddr);

    rewriter.setInsertionPointAfter(loop);

    auto llvmCall = rewriter.create<LLVM::CallOp>(
        loc, readvFunc, ValueRange{fdI32, iovecArrayPtr, vectorCountI32});

    rewriter.replaceOp(op, llvmCall.getResult());
    return success();
  }
};

// ============================================================================
// New Lowerings (Sendfile, Mmap, Prefetch, Submit, Wait)
// ============================================================================

struct SendfileLowering : public ConvertOpToLLVMPattern<io::SendfileOp> {
    using ConvertOpToLLVMPattern<io::SendfileOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(io::SendfileOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto module = op->getParentOfType<ModuleOp>();
        auto *ctx = getContext();

        auto i32Ty = IntegerType::get(ctx, 32);
        auto i64Ty = IntegerType::get(ctx, 64);
        auto ptrTy = LLVM::LLVMPointerType::get(ctx);
        
        // ssize_t sendfile(int out_fd, int in_fd, off_t *offset, size_t count);
        auto funcTy = LLVM::LLVMFunctionType::get(i64Ty, {i32Ty, i32Ty, ptrTy, i64Ty});
        
        // Ensure function exists without triggering unused variable warnings
        getOrInsertFunc(rewriter, module, "sendfile", funcTy);
        
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op, TypeRange{i64Ty}, SymbolRefAttr::get(ctx, "sendfile"), adaptor.getOperands());
            
        return success();
    }
};

struct MmapLowering : public ConvertOpToLLVMPattern<io::MmapOp> {
    using ConvertOpToLLVMPattern<io::MmapOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(io::MmapOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto module = op->getParentOfType<ModuleOp>();
        auto *ctx = getContext();
        auto loc = op.getLoc();

        auto i32Ty = IntegerType::get(ctx, 32);
        auto i64Ty = IntegerType::get(ctx, 64);
        auto ptrTy = LLVM::LLVMPointerType::get(ctx);
        
        // void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
        auto funcTy = LLVM::LLVMFunctionType::get(ptrTy, {ptrTy, i64Ty, i32Ty, i32Ty, i32Ty, i64Ty});
        getOrInsertFunc(rewriter, module, "mmap", funcTy);

        // POSIX mappings: PROT_READ = 1, MAP_PRIVATE = 2
        Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrTy);
        Value protRead = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(1));
        Value mapPrivate = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(2));
        
        SmallVector<Value> args = {
            nullPtr, adaptor.getSize(), protRead, mapPrivate, adaptor.getFd(), adaptor.getOffset()
        };
        
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op, TypeRange{ptrTy}, SymbolRefAttr::get(ctx, "mmap"), args);
            
        return success();
    }
};

struct PrefetchLowering : public ConvertOpToLLVMPattern<io::PrefetchOp> {
    using ConvertOpToLLVMPattern<io::PrefetchOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(io::PrefetchOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto module = op->getParentOfType<ModuleOp>();
        auto *ctx = getContext();
        auto loc = op.getLoc();

        auto i32Ty = IntegerType::get(ctx, 32);
        auto i64Ty = IntegerType::get(ctx, 64);
        
        // int posix_fadvise(int fd, off_t offset, off_t len, int advice);
        auto funcTy = LLVM::LLVMFunctionType::get(i32Ty, {i32Ty, i64Ty, i64Ty, i32Ty});
        getOrInsertFunc(rewriter, module, "posix_fadvise", funcTy);

        // POSIX mapping: POSIX_FADV_WILLNEED = 3
        Value zeroOffset = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, rewriter.getI64IntegerAttr(0));
        Value advice = rewriter.create<LLVM::ConstantOp>(loc, i32Ty, rewriter.getI32IntegerAttr(3));
        
        SmallVector<Value> args = {
            adaptor.getFd(), zeroOffset, adaptor.getLookaheadSize(), advice
        };
        
        // CRITICAL FIX: io.prefetch returns nothing, so we can't "replace" it with an i32.
        // We create the call independently, then erase the prefetch op.
        rewriter.create<LLVM::CallOp>(
            loc, TypeRange{i32Ty}, SymbolRefAttr::get(ctx, "posix_fadvise"), args);
        
        rewriter.eraseOp(op);
            
        return success();
    }
};

struct SubmitLowering : public ConvertOpToLLVMPattern<io::SubmitOp> {
    using ConvertOpToLLVMPattern<io::SubmitOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(io::SubmitOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto module = op->getParentOfType<ModuleOp>();
        auto loc = op.getLoc();

        auto readType = LLVM::LLVMFunctionType::get(
            rewriter.getI64Type(),
            {rewriter.getI32Type(), LLVM::LLVMPointerType::get(rewriter.getContext()), rewriter.getI64Type()}
        );
        getOrInsertFunc(rewriter, module, "read", readType);

        // CRITICAL FIX: Handle MemRefs safely for submit buffers!
        Value rawPtr;
        if (auto memrefType = mlir::dyn_cast<MemRefType>(op.getBuffer().getType())) {
            rawPtr = getStridedElementPtr(rewriter, loc, memrefType, adaptor.getBuffer(), {});
        } else {
            rawPtr = adaptor.getBuffer();
        }

        auto llvmCall = rewriter.create<LLVM::CallOp>(
            loc, TypeRange{rewriter.getI64Type()}, SymbolRefAttr::get(rewriter.getContext(), "read"), 
            ValueRange{adaptor.getFd(), rawPtr, adaptor.getSize()}
        );

        // The ticket is returned as an i32, so truncate the i64 bytes read.
        rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, rewriter.getI32Type(), llvmCall.getResult());
        return success();
    }
};

struct WaitLowering : public ConvertOpToLLVMPattern<io::WaitOp> {
    using ConvertOpToLLVMPattern<io::WaitOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(io::WaitOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // Fallback: The read was already executed synchronously in Submit. 
        // We just cast the ticket (i32) back to bytes read (i64).
        rewriter.replaceOpWithNewOp<LLVM::SExtOp>(op, rewriter.getI64Type(), adaptor.getTicket());
        return success();
    }
};

// ============================================================================
// The Pass Registration
// ============================================================================

struct ConvertIOToLLVMPass : public PassWrapper<ConvertIOToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertIOToLLVMPass)

  llvm::StringRef getArgument() const final { return "convert-io-to-llvm"; }
  llvm::StringRef getDescription() const final { return "Lowers the IO dialect to LLVM IR"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>(); 
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    mlir::io::bootstrapTargetInfo(module);

    MLIRContext *context = &getContext();
    LLVMTypeConverter typeConverter(context);

    ConversionTarget target(*context);
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<arith::ArithDialect>(); 
    target.addLegalOp<io::IOCastOp>();

    // Declare all IO dialect operations illegal to enforce lowering
    target.addIllegalOp<io::BatchWriteOp>();
    target.addIllegalOp<io::BatchWriteVOp>();
    target.addIllegalOp<io::BatchReadOp>();
    target.addIllegalOp<io::BatchReadVOp>();
    target.addIllegalOp<io::SendfileOp>();
    target.addIllegalOp<io::MmapOp>();
    target.addIllegalOp<io::PrefetchOp>();
    target.addIllegalOp<io::SubmitOp>();
    target.addIllegalOp<io::WaitOp>();

    RewritePatternSet patterns(context);
    
    // Existing patterns
    patterns.add<BatchWriteLowering>(typeConverter);
    patterns.add<BatchWriteVLowering>(typeConverter);
    patterns.add<BatchReadLowering>(typeConverter);
    patterns.add<BatchReadVLowering>(typeConverter); 
    
    // New Advanced I/O patterns
    patterns.add<SendfileLowering>(typeConverter);
    patterns.add<MmapLowering>(typeConverter);
    patterns.add<PrefetchLowering>(typeConverter);
    patterns.add<SubmitLowering>(typeConverter);
    patterns.add<WaitLowering>(typeConverter);

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
