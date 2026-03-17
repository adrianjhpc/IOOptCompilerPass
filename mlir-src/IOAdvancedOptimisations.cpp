#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;

namespace {

struct PromoteToZeroCopyPattern : public RewritePattern {
    PromoteToZeroCopyPattern(MLIRContext *context)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/2, context) {}

   LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        auto readCall = dyn_cast<CallOpInterface>(op);
        if (!readCall) return failure();

        auto calleeAttr = dyn_cast<SymbolRefAttr>(readCall.getCallableForCallee());
        if (!calleeAttr || calleeAttr.getRootReference() != "read") return failure();
        
        if (readCall.getArgOperands().size() != 3) return failure();
        
        llvm::errs() << "[IOOpt-Telemetry] Found 'read' call. Scanning block...\n";
        
        Value fdIn = readCall.getArgOperands()[0];
        Value readSize = readCall.getArgOperands()[2];

        // Hardened to trace through array indexing and pointer math
        auto getRootAllocation = [](Value v) {
            while (Operation *def = v.getDefiningOp()) {
                StringRef name = def->getName().getStringRef();
                // ADDED "cir.get_element" so we can detect buffer[index] mutations!
                if (name == "cir.cast" || name == "cir.load" || 
                    name == "cir.ptr_stride" || name == "cir.get_element") {
                    v = def->getOperand(0);
                } else {
                    break;
                }
            }
            return v;
        };        

        Value readBufferRoot = getRootAllocation(readCall.getArgOperands()[1]);

        CallOpInterface writeCall = nullptr;
        
        for (Operation *nextOp = op->getNextNode(); nextOp != nullptr; nextOp = nextOp->getNextNode()) {
            
            if (auto maybeWrite = dyn_cast<CallOpInterface>(nextOp)) {
                auto nextCallee = dyn_cast<SymbolRefAttr>(maybeWrite.getCallableForCallee());
                if (nextCallee && nextCallee.getRootReference() == "write") {
                    if (maybeWrite.getArgOperands().size() == 3) {
                        Value writeBufferRoot = getRootAllocation(maybeWrite.getArgOperands()[1]);
                        if (readBufferRoot == writeBufferRoot) {
                            writeCall = maybeWrite;
                            break; 
                        } else {
                            llvm::errs() << "[IOOpt-Telemetry] Abort: Found 'write', but buffer roots mismatch!\n";
                        }
                    }
                }
            }

            StringRef opName = nextOp->getName().getStringRef();
            
            if (opName == "cir.store") {
                Value storePtr = nextOp->getOperand(1);
                if (getRootAllocation(storePtr) == readBufferRoot) {
                    llvm::errs() << "[IOOpt-Telemetry] Abort: cir.store mutated the buffer root!\n";
                    return failure(); 
                }
                continue; 
            }
            
            if (isa<CallOpInterface>(nextOp)) {
                llvm::errs() << "[IOOpt-Telemetry] Abort: Intervening CallOp found: " << opName << "\n";
                return failure(); 
            }
            
            if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextOp)) {
                if (memInterface.hasEffect<MemoryEffects::Write>()) {
                    llvm::errs() << "[IOOpt-Telemetry] Abort: Intervening memory write found: " << opName << "\n";
                    return failure();
                }
            }
        }

        if (!writeCall) {
            llvm::errs() << "[IOOpt-Telemetry] Abort: Reached end of block. No matching 'write' found.\n";
            return failure();
        }

        Value fdOut = writeCall.getArgOperands()[0];
        Value writeSize = writeCall.getArgOperands()[2];

        if (getRootAllocation(readSize) != getRootAllocation(writeSize)) {
            llvm::errs() << "[IOOpt-Telemetry] Abort: Size variables do not match.\n";
            return failure();
        }

        // The Rewrite
        rewriter.setInsertionPoint(writeCall);
        Value nullOffset = readCall.getArgOperands()[1]; 

        auto sendfileCall = rewriter.create<func::CallOp>(
            writeCall.getLoc(),
            "sendfile",
            readCall->getResultTypes(),
            ValueRange{fdOut, fdIn, nullOffset, writeSize}
        );

        rewriter.replaceOp(writeCall, sendfileCall.getResults());
        rewriter.replaceOp(readCall, sendfileCall.getResults());

        llvm::errs() << "[IOOpt-Telemetry] SUCCESS: Replaced read/write with sendfile!\n";
        return success();
    }

};

struct ZeroCopyPromotionPass : public PassWrapper<ZeroCopyPromotionPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZeroCopyPromotionPass)
    StringRef getArgument() const final { return "io-zero-copy-promotion"; }
    StringRef getDescription() const final { return "Promotes read/write pairs to zero-copy sendfile syscalls"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<func::FuncDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();

        // PRE-PASS: Walk the module sequentially to find a 'read' call.
        // We will steal its exact ClangIR types so our sendfile signature matches perfectly!
        CallOpInterface firstRead = nullptr;
        module.walk([&](CallOpInterface call) {
            auto callee = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
            if (callee && callee.getRootReference() == "read" && call.getArgOperands().size() == 3) {
                firstRead = call;
                return WalkResult::interrupt(); // Stop walking, we got what we need!
            }
            return WalkResult::advance();
        });

        // Safely declare 'sendfile' using the stolen types
        auto sendfileSym = StringAttr::get(context, "sendfile");
        if (firstRead && !module.lookupSymbol(sendfileSym)) {
            OpBuilder builder(module.getBodyRegion());
            
            // Steal the precise ClangIR types dynamically!
            Type fdType = firstRead.getArgOperands()[0].getType();
            Type ptrType = firstRead.getArgOperands()[1].getType(); 
            Type sizeType = firstRead.getArgOperands()[2].getType();
            Type retType = firstRead->getResultTypes()[0];
            
            auto sendfileType = builder.getFunctionType({fdType, fdType, ptrType, sizeType}, {retType});
            builder.create<func::FuncOp>(module.getLoc(), "sendfile", sendfileType).setPrivate();
        }

        // Now run the multithreaded greedy pattern matcher
        RewritePatternSet patterns(context);
        patterns.add<PromoteToZeroCopyPattern>(context);
        
        if (failed(applyPatternsGreedily(module, std::move(patterns))))
            signalPassFailure();
    }
};

// ============================================================================
// 2. Straight-Line Serialization -> Basic Block Vectored I/O
// ============================================================================
struct BlockVectoredIOPattern : public RewritePattern {
    BlockVectoredIOPattern(MLIRContext *context)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        // TODO:
        // 1. Match a 'write' operation.
        // 2. Scan the current Basic Block for subsequent 'write' ops to the SAME file descriptor.
        // 3. Ensure no intervening operations mutate the buffers or branch away.
        // 4. Gather all buffer pointers and lengths.
        // 5. Replace the sequence with an 'io.batch_writev' (iovec array) op.
        return failure();
    }
};

struct BlockVectoredIOPass : public PassWrapper<BlockVectoredIOPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlockVectoredIOPass)
    StringRef getArgument() const final { return "io-block-vectored"; }
    StringRef getDescription() const final { return "Batches straight-line sequential writes into writev"; }

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<BlockVectoredIOPattern>(&getContext());
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

// ============================================================================
// 3. Compute-Bound Block -> Auto-Asynchrony (io_uring / aio)
// ============================================================================
struct PromoteToAsyncIOPass : public PassWrapper<PromoteToAsyncIOPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteToAsyncIOPass)
    StringRef getArgument() const final { return "io-async-promotion"; }
    StringRef getDescription() const final { return "Software pipelines blocking I/O with independent compute"; }

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        // TODO: This is an analysis-heavy pass (not a simple pattern match).
        // 1. Walk the function to find blocking 'read' calls.
        // 2. Perform Data Dependency Analysis on the read buffer.
        // 3. Hoist independent compute operations above the first buffer usage.
        // 4. Replace 'read' with 'io.uring_submit' and inject 'io.uring_wait' right before the buffer is actually used.
    }
};

// ============================================================================
// 4. Bulk Random Access -> Auto-mmap Promotion
// ============================================================================
struct PromoteToMmapPattern : public RewritePattern {
    PromoteToMmapPattern(MLIRContext *context)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/2, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        // TODO:
        // 1. Match an allocation (malloc/alloca) whose size exactly matches a file's size (fstat).
        // 2. Match a single massive 'read' that fills this buffer.
        // 3. Replace the allocation and read with an 'io.mmap' operation.
        return failure();
    }
};

struct MmapPromotionPass : public PassWrapper<MmapPromotionPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MmapPromotionPass)
    StringRef getArgument() const final { return "io-mmap-promotion"; }
    StringRef getDescription() const final { return "Promotes bulk file reads into memory mapped (mmap) buffers"; }

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<PromoteToMmapPattern>(&getContext());
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

// ============================================================================
// 5. Predictable Scan -> Automated Prefetch Injection
// ============================================================================
struct InjectPrefetchPattern : public OpRewritePattern<scf::ForOp> {
    InjectPrefetchPattern(MLIRContext *context)
        : OpRewritePattern<scf::ForOp>(context, /*benefit=*/1) {}

    LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const override {
        // TODO:
        // 1. Check if the loop contains a sequential 'read' pattern (pointer advances by fixed size).
        // 2. Check if the loop also contains heavy compute (high instruction count).
        // 3. Calculate a safe lookahead distance.
        // 4. Inject 'posix_fadvise' (or a custom io.prefetch op) into the loop to trigger kernel read-ahead.
        return failure();
    }
};

struct PrefetchInjectionPass : public PassWrapper<PrefetchInjectionPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefetchInjectionPass)
    StringRef getArgument() const final { return "io-prefetch-injection"; }
    StringRef getDescription() const final { return "Injects kernel read-ahead hints into compute-heavy sequential I/O loops"; }

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<InjectPrefetchPattern>(&getContext());
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

} // end anonymous namespace

// ============================================================================
// Registration Hooks
// ============================================================================
namespace mlir {
namespace io {
    void registerAdvancedIOPasses() {
        PassRegistration<ZeroCopyPromotionPass>();
        PassRegistration<BlockVectoredIOPass>();
        PassRegistration<PromoteToAsyncIOPass>();
        PassRegistration<MmapPromotionPass>();
        PassRegistration<PrefetchInjectionPass>();
    }
} // namespace io
} // namespace mlir
