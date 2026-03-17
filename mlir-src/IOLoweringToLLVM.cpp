#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h" // NEEDED for MemRef to LLVM conversion
#include "mlir/Transforms/DialectConversion.h"        // NEEDED for applyPartialConversion
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "IODialect.h"
#include "TargetUtils.h"

using namespace mlir;

namespace {

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
    Value fdI32 = LLVM::TruncOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), adaptor.getFd());

    // Safely handle both MemRefs and raw LLVM Pointers
    Value rawPtr;
    if (auto memrefType = mlir::dyn_cast<MemRefType>(op.getBuffer().getType())) {
        // It's a MemRef descriptor: extract the raw contiguous pointer
        rawPtr = getStridedElementPtr(rewriter, op.getLoc(), memrefType, adaptor.getBuffer(), {});
    } else {
        // It's already a raw pointer (from C/C++ frontend)
        rawPtr = adaptor.getBuffer();
    }

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

struct BatchWriteVLowering : public ConvertOpToLLVMPattern<io::BatchWriteVOp> {
  using ConvertOpToLLVMPattern<io::BatchWriteVOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(io::BatchWriteVOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();

    // Define the LLVM Types for `struct iovec { void *iov_base; size_t iov_len; }`
    Type voidPtrTy = LLVM::LLVMPointerType::get(ctx);
    Type sizeTy = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();
    Type iovecTy = LLVM::LLVMStructType::getLiteral(ctx, {voidPtrTy, sizeTy});

    // Ensure the OS `writev` function is declared in the module
    auto writevFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("writev");
    if (!writevFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto writevType = LLVM::LLVMFunctionType::get(
          sizeTy, {i32Ty, voidPtrTy, i32Ty});
      writevFunc = rewriter.create<LLVM::LLVMFuncOp>(loc, "writev", writevType);
    }

    Value fdI32 = adaptor.getFd();
    
    Value vectorCountI32 = rewriter.create<LLVM::TruncOp>(loc, i32Ty, adaptor.getCount());

    // Allocate the array of iovec structs on the stack
    Value iovecArrayPtr = rewriter.create<LLVM::AllocaOp>(
        loc, voidPtrTy, iovecTy, vectorCountI32, /*alignment=*/8);

    // Generate a loop to populate the iovec array
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value countIndex = op.getCount(); 

    auto loop = rewriter.create<scf::ForOp>(loc, zero, countIndex, one);
    
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value ivI64 = rewriter.create<arith::IndexCastOp>(loc, sizeTy, iv);

    // Load the pointer address as an i64
    Value ptrValI64 = rewriter.create<memref::LoadOp>(loc, op.getPtrs(), ValueRange{iv});
    
    // Explicitly cast the i64 back into an LLVM Pointer!
    Value ptrVal = rewriter.create<LLVM::IntToPtrOp>(loc, voidPtrTy, ptrValI64);
    
    // Load the size as normal
    Value sizeVal = rewriter.create<memref::LoadOp>(loc, op.getSizes(), ValueRange{iv});

    // Get reference to iovecArray[iv]
    Value iovAddr = rewriter.create<LLVM::GEPOp>(loc, voidPtrTy, iovecTy, iovecArrayPtr, ivI64);

    // Store ptr into iovecArray[iv].iov_base (Index 0)
    Value iovBaseAddr = rewriter.create<LLVM::GEPOp>(
        loc, voidPtrTy, iovecTy, iovAddr, ArrayRef<LLVM::GEPArg>{0, 0});
    rewriter.create<LLVM::StoreOp>(loc, ptrVal, iovBaseAddr);

    // Store size into iovecArray[iv].iov_len (Index 1)
    Value iovLenAddr = rewriter.create<LLVM::GEPOp>(
        loc, voidPtrTy, iovecTy, iovAddr, ArrayRef<LLVM::GEPArg>{0, 1});
    rewriter.create<LLVM::StoreOp>(loc, sizeVal, iovLenAddr);

    rewriter.setInsertionPointAfter(loop);

    // Make the single, optimized system call
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

    // Ensure standard POSIX `read(int fd, void* buf, size_t count)` exists
    auto readFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("read");
    if (!readFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto readType = LLVM::LLVMFunctionType::get(
          rewriter.getI64Type(),
          {rewriter.getI32Type(), LLVM::LLVMPointerType::get(rewriter.getContext()), rewriter.getI64Type()}
      );
      readFunc = LLVM::LLVMFuncOp::create(rewriter, op.getLoc(), "read", readType);
    }

    Value fdI32 = LLVM::TruncOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), adaptor.getFd());
    
    // Safely handle both MemRefs and raw LLVM Pointers
    Value rawPtr;
    if (auto memrefType = mlir::dyn_cast<MemRefType>(op.getBuffer().getType())) {
        // It's a MemRef descriptor: extract the raw contiguous pointer
        rawPtr = getStridedElementPtr(rewriter, op.getLoc(), memrefType, adaptor.getBuffer(), {});
    } else {
        // It's already a raw pointer (from C/C++ frontend)
        rawPtr = adaptor.getBuffer();
    }

    auto llvmCall = LLVM::CallOp::create(
        rewriter,
        op.getLoc(),
        readFunc,
        ValueRange{fdI32, rawPtr, adaptor.getTotalSize()}
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

    // 1. Define LLVM Types (iovec { void*, size_t })
    Type voidPtrTy = LLVM::LLVMPointerType::get(ctx);
    Type sizeTy = rewriter.getI64Type();
    Type i32Ty = rewriter.getI32Type();
    Type iovecTy = LLVM::LLVMStructType::getLiteral(ctx, {voidPtrTy, sizeTy});

    // 2. Ensure `readv` is declared (ssize_t readv(int fd, const struct iovec *iov, int iovcnt))
    auto readvFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("readv");
    if (!readvFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto readvType = LLVM::LLVMFunctionType::get(sizeTy, {i32Ty, voidPtrTy, i32Ty});
      readvFunc = rewriter.create<LLVM::LLVMFuncOp>(loc, "readv", readvType);
    }

    // 3. Prepare arguments
    Value fdI32 = adaptor.getFd();
    Value vectorCountI32 = rewriter.create<LLVM::TruncOp>(loc, i32Ty, adaptor.getCount());

    // 4. Allocate iovec array on stack
    Value iovecArrayPtr = rewriter.create<LLVM::AllocaOp>(
        loc, voidPtrTy, iovecTy, vectorCountI32, /*alignment=*/8);

    // 5. Setup loop to populate iovec array
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value countIndex = op.getCount(); 

    auto loop = rewriter.create<scf::ForOp>(loc, zero, countIndex, one);
    
    rewriter.setInsertionPointToStart(loop.getBody());
    Value iv = loop.getInductionVar();
    Value ivI64 = rewriter.create<arith::IndexCastOp>(loc, sizeTy, iv);

    // --- Synchronized MemRef Access (Same as WriteV) ---
    // Load the pointer address (i64) and convert to LLVM Pointer
    Value ptrValI64 = rewriter.create<memref::LoadOp>(loc, op.getPtrs(), ValueRange{iv});
    Value ptrVal = rewriter.create<LLVM::IntToPtrOp>(loc, voidPtrTy, ptrValI64);
    
    // Load the size (i64)
    Value sizeVal = rewriter.create<memref::LoadOp>(loc, op.getSizes(), ValueRange{iv});

    // 6. Populate iovec[iv]
    Value iovAddr = rewriter.create<LLVM::GEPOp>(loc, voidPtrTy, iovecTy, iovecArrayPtr, ivI64);

    // iovec[iv].iov_base
    Value iovBaseAddr = rewriter.create<LLVM::GEPOp>(
        loc, voidPtrTy, iovecTy, iovAddr, ArrayRef<LLVM::GEPArg>{0, 0});
    rewriter.create<LLVM::StoreOp>(loc, ptrVal, iovBaseAddr);

    // iovec[iv].iov_len
    Value iovLenAddr = rewriter.create<LLVM::GEPOp>(
        loc, voidPtrTy, iovecTy, iovAddr, ArrayRef<LLVM::GEPArg>{0, 1});
    rewriter.create<LLVM::StoreOp>(loc, sizeVal, iovLenAddr);

    rewriter.setInsertionPointAfter(loop);

    // 7. Make the system call
    auto llvmCall = rewriter.create<LLVM::CallOp>(
        loc, readvFunc, ValueRange{fdI32, iovecArrayPtr, vectorCountI32});

    rewriter.replaceOp(op, llvmCall.getResult());
    
    return success();
  }
};

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

    target.addIllegalOp<io::BatchWriteOp>();
    target.addIllegalOp<io::BatchWriteVOp>();
    target.addIllegalOp<io::BatchReadOp>();
    target.addIllegalOp<io::BatchReadVOp>();

    RewritePatternSet patterns(context);
    patterns.add<BatchWriteLowering>(typeConverter);
    patterns.add<BatchWriteVLowering>(typeConverter);
    patterns.add<BatchReadLowering>(typeConverter);
    patterns.add<BatchReadVLowering>(typeConverter); 

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
