#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Analysis/AliasAnalysis.h"

#include "IODialect.h"

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

        auto sendfileOp = mlir::io::SendfileOp::create(
            rewriter,
            writeCall.getLoc(),
            readCall->getResult(0).getType(), // Return type (bytes transferred)
            fdOut,                           // out_fd
            fdIn,                            // in_fd
            nullOffset,                      // offset (buffer pointer)
            writeSize                        // count
        );

        rewriter.replaceOp(writeCall, sendfileOp.getBytesWritten());
        rewriter.replaceOp(readCall, sendfileOp.getBytesWritten());

        llvm::errs() << "[IOOpt-Telemetry] SUCCESS: Replaced read/write with sendfile!\n";
        return success();
    }

};

struct ZeroCopyPromotionPass : public PassWrapper<ZeroCopyPromotionPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZeroCopyPromotionPass)
    StringRef getArgument() const final { return "io-zero-copy-promotion"; }
    StringRef getDescription() const final { return "Promotes read/write pairs to zero-copy io.sendfile ops"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mlir::io::IODialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();

        RewritePatternSet patterns(context);
        patterns.add<PromoteToZeroCopyPattern>(context);
        
        if (failed(applyPatternsGreedily(module, std::move(patterns))))
            signalPassFailure();
    }
};

// ============================================================================
// 2. Straight-Line Serialization -> Basic Block Vectored I/O (Deterministic)
// ============================================================================
struct BlockVectoredIOPass : public PassWrapper<BlockVectoredIOPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlockVectoredIOPass)

    StringRef getArgument() const final { return "io-block-vectored"; }
    StringRef getDescription() const final { return "Batches straight-line sequential writes into writev"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<func::FuncDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();

        // --------------------------------------------------------------------
        // Step 1: PRE-PASS (Steal types and inject ioopt_writev_* intrinsics)
        // --------------------------------------------------------------------
        CallOpInterface firstWrite = nullptr;
        module.walk([&](CallOpInterface call) {
            auto callee = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
            if (callee && callee.getRootReference() == "write" && call.getArgOperands().size() == 3) {
                firstWrite = call;
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });

        // If there are no writes in this module, just exit early!
        if (!firstWrite) return;

        OpBuilder builder(module.getBodyRegion());
        Type fdType = firstWrite.getArgOperands()[0].getType();
        Type ptrType = firstWrite.getArgOperands()[1].getType(); 
        Type sizeType = firstWrite.getArgOperands()[2].getType();
        Type retType = firstWrite->getResultTypes()[0];
        
        for (int i = 2; i <= 4; ++i) {
            std::string funcName = "ioopt_writev_" + std::to_string(i);
            if (!module.lookupSymbol(funcName)) {
                SmallVector<Type> argTypes;
                argTypes.push_back(fdType);
                for(int j = 0; j < i; j++) {
                    argTypes.push_back(ptrType);
                    argTypes.push_back(sizeType);
                }
                auto funcType = builder.getFunctionType(argTypes, {retType});
                func::FuncOp::create(builder, module.getLoc(), funcName, funcType).setPrivate();
            }
        }

        // --------------------------------------------------------------------
        // Step 2: DETERMINISTIC TOP-DOWN WALK
        // --------------------------------------------------------------------
        // We bypass the greedy pattern rewriter entirely to guarantee order.
        SmallVector<SmallVector<Operation*, 4>> allBatches;

        module.walk([&](Block *block) {
            SmallVector<Operation*, 4> currentBatch;
            Value currentFdRoot = nullptr;

            // Our trusty SSA Root Tracer
            auto getRootAllocation = [](Value v) {
                while (Operation *def = v.getDefiningOp()) {
                    StringRef name = def->getName().getStringRef();
                    if (name == "cir.load" || name == "cir.cast" || 
                        name == "cir.get_element" || name == "cir.ptr_stride") {
                        v = def->getOperand(0);
                    } else {
                        break;
                    }
                }
                return v;
            };

            for (Operation &opRef : *block) {
                Operation *op = &opRef;
                
                // If we find a write, check if we can add it to the current batch
                if (auto call = dyn_cast<CallOpInterface>(op)) {
                    auto callee = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
                    if (callee && callee.getRootReference() == "write" && call.getArgOperands().size() == 3) {
                        Value fdRoot = getRootAllocation(call.getArgOperands()[0]);
                        
                        if (currentBatch.empty()) {
                            currentBatch.push_back(op);
                            currentFdRoot = fdRoot;
                        } else if (fdRoot == currentFdRoot) {
                            currentBatch.push_back(op);
                            if (currentBatch.size() == 4) { // Cap at 4 for intrinsic chunking
                                allBatches.push_back(currentBatch);
                                currentBatch.clear();
                                currentFdRoot = nullptr;
                            }
                        } else {
                            // Hit a write to a DIFFERENT file. Save old batch, start new one.
                            if (currentBatch.size() >= 2) allBatches.push_back(currentBatch);
                            currentBatch.clear();
                            currentBatch.push_back(op);
                            currentFdRoot = fdRoot;
                        }
                        continue;
                    }
                }

                StringRef opName = op->getName().getStringRef();
                
                // Safe memory preparation instructions (ignore them)
                if (opName == "cir.load" || opName == "cir.cast" || 
                    opName == "cir.get_element" || opName == "cir.ptr_stride" ||
                    opName == "cir.const") {
                    continue;
                }

                // HAZARD DETECTED: Bank the current batch and reset!
                if (currentBatch.size() >= 2) allBatches.push_back(currentBatch);
                currentBatch.clear();
                currentFdRoot = nullptr;
            }
            
            // End of the block: Bank any remaining writes
            if (currentBatch.size() >= 2) allBatches.push_back(currentBatch);
        });

        // --------------------------------------------------------------------
        // Step 3: APPLY THE REPLACEMENTS
        // --------------------------------------------------------------------
        // We do this AFTER the walk to avoid crashing the block iterator!
        for (auto &batch : allBatches) {
            // Insert exactly where the last write was
            OpBuilder replaceBuilder(batch.back()); 
            std::string funcName = "ioopt_writev_" + std::to_string(batch.size());
            
            SmallVector<Value, 9> newArgs;
            auto firstWriteInBatch = cast<CallOpInterface>(batch[0]);
            
            newArgs.push_back(firstWriteInBatch.getArgOperands()[0]); // Shared FD
            
            for (Operation *w : batch) {
                auto wCall = cast<CallOpInterface>(w);
                newArgs.push_back(wCall.getArgOperands()[1]); // Buffer
                newArgs.push_back(wCall.getArgOperands()[2]); // Size
            }

            auto writevCall = func::CallOp::create(
                replaceBuilder,
                batch.back()->getLoc(),
                funcName,
                firstWriteInBatch->getResultTypes(),
                newArgs
            );

            // Replace all original writes with the new writev call
            for (Operation *w : batch) {
                w->replaceAllUsesWith(writevCall.getResults());
                w->erase();
            }
            
            llvm::errs() << "[IOOpt-Vectored] SUCCESS: Merged " << batch.size() << " writes into " << funcName << "!\n";
        }
    }
};

// ============================================================================
// 3. Compute-Bound Block -> Auto-Asynchrony (io_uring / aio)
// ============================================================================

namespace {

struct PromoteToAsyncIOPass : public PassWrapper<PromoteToAsyncIOPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteToAsyncIOPass)
    StringRef getArgument() const final { return "io-async-promotion"; }
    StringRef getDescription() const final { return "Software pipelines blocking I/O with independent compute"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mlir::io::IODialect>();
    }

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        IRRewriter rewriter(&getContext());
        
        // Request MLIR's Alias Analysis for the current function
        AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();

        SmallVector<func::CallOp> readCandidates;

        // alk the function to find synchronous 'read' calls
        func.walk([&](func::CallOp callOp) {
            StringRef callee = callOp.getCallee();
            if (callee == "read" || callee == "read64" || callee == "read32") {
                readCandidates.push_back(callOp);
            }
        });

        for (func::CallOp readOp : readCandidates) {
            if (readOp.getNumOperands() < 3) continue;
            
            Value fd = readOp.getOperand(0);
            Value buffer = readOp.getOperand(1);
            Value size = readOp.getOperand(2);
            Value bytesReadResult = readOp.getResult(0);

            Block *block = readOp->getBlock();
            auto it = Block::iterator(readOp);
            ++it; // Start scanning immediately *after* the read

            Operation *waitInsertionPoint = nullptr;
            int independentComputeCount = 0;

            // The Dependency Scanner (Now with Alias Analysis!)
            while (it != block->end()) {
                Operation &currentOp = *it;
                bool isDependent = false;

                // Rule A: Does this operation use the bytes_read integer result directly?
                for (Value operand : currentOp.getOperands()) {
                    if (operand == bytesReadResult) {
                        isDependent = true;
                        break;
                    }
                }

                if (!isDependent) {
                    // Rule B: Ask AliasAnalysis if this instruction touches our buffer!
                    // In Async I/O, the Kernel owns the buffer between submit and wait.
                    // If the CPU tries to Read (Ref) OR Write (Mod) to it, we must wait.
                    ModRefResult modRef = aliasAnalysis.getModRef(&currentOp, buffer);
                    
                    if (modRef.isMod() || modRef.isRef()) {
                        isDependent = true;
                    }
                }

                // Rule C: Safely handle control flow boundaries and black boxes
                if (!isDependent) {
                    // If it's a terminator (branch, return), we must stop and wait.
                    // We cannot safely float an async wait into another basic block 
                    // without advanced control-flow dominance analysis.
                    if (currentOp.hasTrait<OpTrait::IsTerminator>()) {
                        isDependent = true;
                    }
                    // If it's an opaque function call that might do hidden I/O or state mutation
                    else if (auto call = dyn_cast<CallOpInterface>(&currentOp)) {
                        auto memEffects = dyn_cast<MemoryEffectOpInterface>(&currentOp);
                        // If it doesn't explicitly declare itself free of memory effects, assume it's a hazard.
                        if (!memEffects || !memEffects.hasNoEffect()) {
                            isDependent = true;
                        }
                    }
                }

                // We found the hazard! This is where the wait() must go.
                if (isDependent) {
                    waitInsertionPoint = &currentOp;
                    break;
                }

                independentComputeCount++;
                ++it;
            }

            // Evaluate Profitability
            // We only split the I/O if we actually managed to jump over independent instructions.
            // (e.g., if independentComputeCount is 0, a synchronous read is faster anyway).
            if (independentComputeCount > 0 && waitInsertionPoint) {
                rewriter.setInsertionPoint(readOp);

                // Create the ASYNC SUBMIT call
                auto submitOp = mlir::io::SubmitOp::create(
                    rewriter, 
                    readOp.getLoc(),
                    rewriter.getI32Type(), // Return ticket type
                    fd, buffer, size
                );

                // Move the rewriter down to right before the hazard
                rewriter.setInsertionPoint(waitInsertionPoint);

                // Create the ASYNC WAIT call
                auto waitOp = mlir::io::WaitOp::create(
                    rewriter, 
                    readOp.getLoc(),
                    readOp.getResult(0).getType(), // Return bytes read
                    submitOp.getTicket()
                );

                // Swap the synchronous return value for the async wait return value
                readOp.replaceAllUsesWith(waitOp->getResults());
                
                // Erase the old synchronous read
                rewriter.eraseOp(readOp);
            }
        }
    }
};

} // end anonymous namespace

// ============================================================================
// 4. Bulk Random Access -> Auto-mmap Promotion
// ============================================================================
struct PromoteToMmapPattern : public RewritePattern {
    PromoteToMmapPattern(MLIRContext *context)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        Value fd, buffer, size;
        
        // 1. Find a read operation
        if (auto readOp = dyn_cast<mlir::io::ReadOp>(op)) {
            fd = readOp.getFd();
            buffer = readOp.getOperand(1);
            size = readOp.getSize();
        }

        else if (auto callOp = dyn_cast<func::CallOp>(op)) {
            StringRef callee = callOp.getCallee(); 
            if (callee == "read" || callee == "read64" || callee == "read32") {
                if (callOp.getNumOperands() != 3) return failure();
                fd = callOp.getOperand(0);
                buffer = callOp.getOperand(1);
                size = callOp.getOperand(2);
            } else {
                return failure();
            }
        }
        else {
            return failure();
        }

        // 2. Trace the buffer pointer back to its allocation source
        Operation *allocOp = buffer.getDefiningOp();
        if (!allocOp) return failure();

        Value allocSize;
        if (auto callAlloc = dyn_cast<func::CallOp>(allocOp)) {
            StringRef callee = callAlloc.getCallee();
            if ((callee == "malloc" || callee == "malloc32") && callAlloc.getNumOperands() == 1) {
                allocSize = callAlloc.getOperand(0);
            } else {
                return failure();
            }
        }

        // 3. Profitability & Safety Check
        // allocSize and size are SSA Values. This strictly ensures the program 
        // used the exact same dynamic variable for both the malloc and the read.
        if (allocSize != size) {
            return failure(); 
        }

        // 4. The Rewrite
        rewriter.setInsertionPoint(allocOp);
        Location loc = allocOp->getLoc();

        Value zeroOffset = arith::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(0));

        Value sizeI64 = size;
        if (!sizeI64.getType().isInteger(64)) {
            sizeI64 = arith::ExtUIOp::create(rewriter, loc, rewriter.getI64Type(), size);
        }

        auto mmapOp = mlir::io::MmapOp::create(
            rewriter,
            loc,
            buffer.getType(), 
            fd,
            sizeI64, 
            zeroOffset
        );

        // Replace all downstream uses of the malloc'd buffer with the mmap'd buffer
        rewriter.replaceOp(allocOp, mmapOp.getBuffer());
        
        Value replacementResult = size;
        Type expectedReturnType = op->getResult(0).getType();
        
        if (replacementResult.getType() != expectedReturnType) {
            if (replacementResult.getType().getIntOrFloatBitWidth() < expectedReturnType.getIntOrFloatBitWidth()) {
                replacementResult = arith::ExtUIOp::create(rewriter, loc, expectedReturnType, replacementResult);
            } else {
                replacementResult = arith::TruncIOp::create(rewriter, loc, expectedReturnType, replacementResult);
            }
        }

        rewriter.replaceOp(op, replacementResult); 

        llvm::errs() << "[IOOpt-Telemetry] SUCCESS: Promoted malloc+read to io.mmap!\n";
        return success();
    }
};

struct MmapPromotionPass : public PassWrapper<MmapPromotionPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MmapPromotionPass)
    StringRef getArgument() const final { return "io-mmap-promotion"; }
    StringRef getDescription() const final { return "Promotes bulk file reads into memory mapped (mmap) buffers"; }

    // CRITICAL: Declare dependencies on io and arith dialects!
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mlir::io::IODialect, mlir::arith::ArithDialect>();
    }

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<PromoteToMmapPattern>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

struct InjectPrefetchPattern : public OpRewritePattern<scf::ForOp> {
    InjectPrefetchPattern(MLIRContext *context)
        : OpRewritePattern<scf::ForOp>(context, /*benefit=*/1) {}

    LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const override {
        Value fdToPrefetch;
        Value sizeToPrefetch;
        Operation *readInsertionPoint = nullptr;
        
        bool alreadyPrefetched = false;
        int computeInstructionCount = 0;

        // 1. Scan the inside of the loop
        forOp.walk([&](Operation *op) {
            // Guard against infinite loops
            if (isa<mlir::io::PrefetchOp>(op)) {
                alreadyPrefetched = true;
            } 
            else if (auto readOp = dyn_cast<mlir::io::ReadOp>(op)) {
                if (!readInsertionPoint) {
                    fdToPrefetch = readOp.getFd();
                    sizeToPrefetch = readOp.getSize();
                    readInsertionPoint = op;
                }
            } 
            else if (auto callOp = dyn_cast<func::CallOp>(op)) {
                StringRef callee = callOp.getCallee(); 
                // Fix: StringRef is an object, not a pointer. Use the == operator!
                if (callee == "read" || callee == "read64" || callee == "read32") {
                    if (!readInsertionPoint && callOp.getNumOperands() == 3) {
                        fdToPrefetch = callOp.getOperand(0);
                        sizeToPrefetch = callOp.getOperand(2);
                        readInsertionPoint = op;
                    }
                }
            } 
            else if (!isa<scf::YieldOp>(op) && !isa<scf::ForOp>(op)) {
                computeInstructionCount++;
            }
        });

        if (!readInsertionPoint) return failure(); 
        if (alreadyPrefetched) return failure();   
        if (computeInstructionCount < 10) return failure(); 

        // 2. Mutate the loop safely using the new MLIR 22 API
        rewriter.modifyOpInPlace(forOp, [&]() {
            rewriter.setInsertionPoint(readInsertionPoint);
            Location loc = readInsertionPoint->getLoc();

            // Mathematically Safe Type Alignment for the math operations
            Value sizeI64 = sizeToPrefetch;
            if (!sizeI64.getType().isInteger(64)) {
                sizeI64 = arith::ExtUIOp::create(rewriter, loc, rewriter.getI64Type(), sizeToPrefetch);
            }

            // Calculate Lookahead (size * 4) using the modern static create methods
            Value four = arith::ConstantOp::create(rewriter, loc, rewriter.getI64IntegerAttr(4));
            Value lookaheadSize = arith::MulIOp::create(rewriter, loc, sizeI64, four);

            mlir::io::PrefetchOp::create(
                rewriter,
                loc,
                fdToPrefetch,
                lookaheadSize
            );
            
            llvm::errs() << "[IOOpt-Telemetry] SUCCESS: Injected io.prefetch ahead of loop read!\n";
        });

        return success();
    }
};

struct PrefetchInjectionPass : public PassWrapper<PrefetchInjectionPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrefetchInjectionPass)
    StringRef getArgument() const final { return "io-prefetch-injection"; }
    StringRef getDescription() const final { return "Injects kernel read-ahead hints into compute-heavy sequential I/O loops"; }

    // CRITICAL: We are generating new operations from the 'io' and 'arith' dialects.
    // We must declare them here so MLIR loads them into the context!
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mlir::io::IODialect, mlir::arith::ArithDialect>();
    }

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<InjectPrefetchPattern>(&getContext());
        
        // Using the modern applyPatternsGreedily!
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
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
