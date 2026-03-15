#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"



#include "IODialect.h"

using namespace mlir;

namespace {

// Helper to extract a constant integer from an MLIR Value
static std::optional<int64_t> getConstantIntValue(Value val) {
    if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>())
        return constOp.value();
    if (auto constIntOp = val.getDefiningOp<arith::ConstantIntOp>())
        return constIntOp.value();
    return std::nullopt;
}

// Proves that: (Offset_Multiplier * Loop_Step) == Write_Size
static bool verifySCEVOffset(Value dynamicOffset, scf::ForOp loop, Value writeSize) {
    Value iv = loop.getInductionVar();
    
    // Attempt to resolve the writeSize and step to compile-time constants
    auto optWriteSize = getConstantIntValue(writeSize);
    auto optStep = getConstantIntValue(loop.getStep());
    
    // If we can't statically prove the step and size, it is too dangerous to 
    // batch contiguously. Conservatively abort.
    if (!optWriteSize || !optStep) return false;
    
    int64_t targetAdvance = *optWriteSize;
    int64_t step = *optStep;

    // Case 1: The offset is the induction variable (Multiplier = 1)
    // Example: ptr = base + iv. 
    // This is only contiguous if the loop step exactly matches the write size.
    if (dynamicOffset == iv) {
        return step == targetAdvance;
    }

    // Case 2: The offset is explicitly calculated: offset = iv * multiplier
    if (auto mulOp = dynamicOffset.getDefiningOp<arith::MulIOp>()) {
        Value lhs = mulOp.getLhs();
        Value rhs = mulOp.getRhs();

        if (lhs == iv || rhs == iv) {
            Value multiplierVal = (lhs == iv) ? rhs : lhs;
            auto optMultiplier = getConstantIntValue(multiplierVal);
            
            if (optMultiplier) {
                return (*optMultiplier * step) == targetAdvance;
            }
        }
    }

    return false;
}

static bool isContiguousMemoryAccess(Value buffer, scf::ForOp loop, Value writeSize, Value &outBasePointer) {
    // If the buffer doesn't change during the loop, it's writing to the exact 
    // same memory address every iteration. This is not contiguous batchable.
    if (loop.isDefinedOutsideOfLoop(buffer)) return false;

    Operation *defOp = buffer.getDefiningOp();
    if (!defOp) return false;

    // ------------------------------------------------------------------
    // PATTERN 1: mlir::memref::SubViewOp
    // Example: %sub = memref.subview %base[%iv] [%size] [%stride]
    // ------------------------------------------------------------------
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
        outBasePointer = subviewOp.getSource();
        
        // Base pointer must live outside the loop
        if (!loop.isDefinedOutsideOfLoop(outBasePointer)) return false;

        auto mixedOffsets = subviewOp.getMixedOffsets();
        if (mixedOffsets.empty()) return false; 

        // Get the first dimension's offset
        OpFoldResult firstOffset = mixedOffsets.front();
         
        // Modern LLVM uses global casting templates instead of member functions.
        // dyn_cast safely checks if it's a Value and extracts it in one step.
        if (auto dynVal = dyn_cast<Value>(firstOffset)) {
            return verifySCEVOffset(dynVal, loop, writeSize);
        }
        
        // If it's not a Value, it's a static constant Attribute. 
        // A static offset means it writes to the exact same memory address 
        // on every single loop iteration. This is not contiguous.
        return false;
    }

    // ------------------------------------------------------------------
    // PATTERN 2: mlir::LLVM::GEPOp (GetElementPtr)
    // Example: %ptr = llvm.getelementptr %base[%iv]
    // ------------------------------------------------------------------
    if (auto gepOp = dyn_cast<LLVM::GEPOp>(defOp)) {
        outBasePointer = gepOp.getBase();
        
        if (!loop.isDefinedOutsideOfLoop(outBasePointer)) return false;

        // GEPs separate static and dynamic indices.
        auto dynamicIndices = gepOp.getDynamicIndices();
        if (dynamicIndices.empty()) return false;

        // Check the primary dynamic index advancing the pointer
        return verifySCEVOffset(dynamicIndices.back(), loop, writeSize);
    }

    // ------------------------------------------------------------------
    // PATTERN 3: Raw arith.addi Pointer Math
    // Example: %ptr = arith.addi %base, %offset
    // ------------------------------------------------------------------
    if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
        Value lhs = addOp.getLhs();
        Value rhs = addOp.getRhs();

        Value dynamicOffset;
        if (loop.isDefinedOutsideOfLoop(lhs)) {
            outBasePointer = lhs;
            dynamicOffset = rhs;
        } else if (loop.isDefinedOutsideOfLoop(rhs)) {
            outBasePointer = rhs;
            dynamicOffset = lhs;
        } else {
            return false; // Neither side is a stable base pointer
        }

        return verifySCEVOffset(dynamicOffset, loop, writeSize);
    }

    // Unrecognized memory access pattern
    return false;
}

struct HoistWriteLoopPattern : public OpRewritePattern<scf::ForOp> {
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

    // Verify FD is stable
    if (!loop.isDefinedOutsideOfLoop(writeOp.getFd())) {
      return failure();
    }

    // Verify Memory Access is Contiguous and extract the stable Base Pointer
    Value basePointer;
    if (!isContiguousMemoryAccess(writeOp.getBuffer(), loop, writeOp.getSize(), basePointer)) {
        // If it's not contiguous, we abort the batching 
        // (This prevents silent memory corruption)
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
    
    // Create the optimized io.batch_write operation using the base pointer, 
    // not the loop-variant buffer pointer!
    rewriter.create<io::BatchWriteOp>(
        loc, 
        rewriter.getI64Type(), 
        writeOp.getFd(), 
        basePointer, 
        totalSize
    );

    // Completely erase the original loop
    rewriter.eraseOp(loop);

    return success();
  }
};

struct HoistReadLoopPattern : public OpRewritePattern<scf::ForOp> {
  AliasAnalysis &aliasAnalysis;

  // Constructor takes the AliasAnalysis engine from the Pass Manager
  HoistReadLoopPattern(MLIRContext *context, AliasAnalysis &aa)
      : OpRewritePattern<scf::ForOp>(context), aliasAnalysis(aa) {}

  LogicalResult matchAndRewrite(scf::ForOp loop, PatternRewriter &rewriter) const override {
    Block *body = loop.getBody();
    if (!body || body->empty()) return failure();

    io::ReadOp readOp = nullptr;

    // Find the ReadOp
    for (Operation &op : *body) {
      if (isa<scf::YieldOp>(op)) continue;
      if (auto ioRead = dyn_cast<io::ReadOp>(op)) {
        if (readOp) return failure(); 
        readOp = ioRead;
      }
    }
    if (!readOp || !loop.isDefinedOutsideOfLoop(readOp.getFd())) return failure();

    // Hazard checking
    for (Operation &op : *body) {
      if (isa<scf::YieldOp, io::ReadOp>(op)) continue;

      // Does this operation write to memory?
      auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
      if (memEffect && memEffect.hasEffect<MemoryEffects::Write>()) {
        SmallVector<MemoryEffects::EffectInstance, 4> effects;
        memEffect.getEffects<MemoryEffects::Write>(effects);
        
        for (auto &effect : effects) {
          Value writtenVal = effect.getValue();
          if (writtenVal) {
            // Ask MLIR's Alias Analysis if the written memory overlaps with our read buffer
            AliasResult aliasResult = aliasAnalysis.alias(readOp.getBuffer(), writtenVal);
            if (!aliasResult.isNo()) {
              // It's a MayAlias or MustAlias. We cannot safely hoist
              return failure(); 
            }
          } else {
            // Opaque memory write (e.g., an external function call). Unsafe to hoist.
            return failure();
          }
        }
      } else if (!isMemoryEffectFree(&op)) {
         // Catch-all for operations that lack specific memory interfaces but aren't pure
         return failure();
      }
    }

    Location loc = loop.getLoc();
    Value diff = rewriter.create<arith::SubIOp>(loc, loop.getUpperBound(), loop.getLowerBound());
    Value tripCount = rewriter.create<arith::DivSIOp>(loc, diff, loop.getStep());

    rewriter.setInsertionPoint(loop);

    // Contiguous vs Vector Routing
    Value basePointer;
    if (isContiguousMemoryAccess(readOp.getBuffer(), loop, readOp.getSize(), basePointer)) {
        // Fast contigious read
        Value totalSize = rewriter.create<arith::MulIOp>(loc, tripCount, readOp.getSize());
        rewriter.create<io::BatchReadOp>(loc, rewriter.getI64Type(), readOp.getFd(), basePointer, totalSize);
    } else {
        // Gather (readv)
        Value tripCountIndex = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), tripCount);

        auto ptrArrayType = MemRefType::get({ShapedType::kDynamic}, LLVM::LLVMPointerType::get(getContext()));
        auto sizeArrayType = MemRefType::get({ShapedType::kDynamic}, rewriter.getI64Type());
        
        Value ptrsMemref = rewriter.create<memref::AllocaOp>(loc, ptrArrayType, tripCountIndex);
        Value sizesMemref = rewriter.create<memref::AllocaOp>(loc, sizeArrayType, tripCountIndex);

        // Build an address-calculation loop before the main loop
        auto calcLoop = rewriter.create<scf::ForOp>(loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep());
        rewriter.setInsertionPointToStart(calcLoop.getBody());

        Value currentIV = calcLoop.getInductionVar();
        Value ivOffset = rewriter.create<arith::SubIOp>(loc, currentIV, loop.getLowerBound());
        Value arrayIdx = rewriter.create<arith::DivSIOp>(loc, ivOffset, loop.getStep());
        Value arrayIdxCast = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), arrayIdx);

        // Clone only the pure address/math calculations
        IRMapping mapping;
        mapping.map(loop.getInductionVar(), currentIV);
        for (Operation &op : *body) {
            if (isa<io::ReadOp, scf::YieldOp>(op)) continue;
            if (isMemoryEffectFree(&op)) {
                rewriter.clone(op, mapping);
            }
        }

        Value mappedBuffer = mapping.lookupOrDefault(readOp.getBuffer());
        Value mappedSize = mapping.lookupOrDefault(readOp.getSize());
        
        rewriter.create<memref::StoreOp>(loc, mappedBuffer, ptrsMemref, arrayIdxCast);
        rewriter.create<memref::StoreOp>(loc, mappedSize, sizesMemref, arrayIdxCast);

        // Emit the batched gather operation immediately after our calculation loop
        rewriter.setInsertionPoint(loop); 
        rewriter.create<io::BatchReadVOp>(loc, readOp.getFd(), ptrsMemref, sizesMemref, tripCount);
    }

    // Clean up the original loop
    // Replace the slow I/O calls inside the loop with just the expected byte count,
    // leaving the processing logic completely intact to run against the now-populated memory
    rewriter.setInsertionPoint(readOp);
    rewriter.replaceOp(readOp, readOp.getSize());

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
    patterns.add<HoistWriteLoopPattern>(context);
    patterns.add<HoistReadLoopPattern>(context);    
 
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
