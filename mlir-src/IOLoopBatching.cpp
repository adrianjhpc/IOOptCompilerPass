#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"

#include "llvm/Support/CommandLine.h"

#include "IODialect.h"
#include "TargetUtils.h"

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
   
    if (auto castOp = dynamicOffset.getDefiningOp<arith::IndexCastOp>()) {
        dynamicOffset = castOp.getIn();
    }
 
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


// Target cir::ForOp directly to bypass any broken interfaces!
struct CirLoopBatchingPattern {
  static LogicalResult matchAndRewrite(cir::ForOp forOp, IRRewriter &rewriter) {    
    Region &bodyRegion = forOp.getBody();
    Region &condRegion = forOp.getCond();

    // 1. Find the I/O call
    cir::CallOp ioCall = nullptr;
    bool isRead = false; 
    bool isPositional = false;
    bool isVFD = false;
    StringRef detectedFuncName = "";

    bodyRegion.walk([&](cir::CallOp call) {
      if (auto calleeAttr = call->getAttrOfType<FlatSymbolRefAttr>("callee")) {
        StringRef funcName = calleeAttr.getValue();
        
        if (funcName == "write" || funcName == "read" ||
            funcName == "pwrite" || funcName == "pread" ||
            funcName == "FileWrite" || funcName == "FileRead") {
            
          ioCall = call;
          detectedFuncName = funcName;
          isRead = funcName.contains("ead"); // Matches read, pread, FileRead
          isPositional = funcName.starts_with("p") || funcName.starts_with("File");
          isVFD = funcName.starts_with("File");
          
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    if (!ioCall) return failure();
    
    llvm::errs() << "\n[IOOpt] Analyzing loop containing: " << detectedFuncName << "\n";

    // --- SAFETY LOCK 1: VFD GUARD ---
    if (isVFD) {
        llvm::errs() << "[IOOpt] ABORT: '" << detectedFuncName << "' uses a PostgreSQL Virtual File Descriptor (VFD).\n";
        llvm::errs() << "        Passing this to a Linux writev() syscall will result in EBADF.\n";
        return failure();
    }

    // --- SAFETY LOCK 2: POSITIONAL OFFSET GUARD & SCEV ---
    Value initialOffsetForSyscall = nullptr;

    if (isPositional) {
        if (ioCall.getNumOperands() < 4) {
            llvm::errs() << "[IOOpt] ABORT: Positional I/O detected but missing offset operand.\n";
            return failure();
        }
        
        Value offsetArg = ioCall.getOperand(3);
        Value sizeArg = ioCall.getOperand(2);

        // Helper: Strip away CIR casts to see the raw underlying value
        auto stripCasts = [](Value val) -> Value {
            while (auto cast = val.getDefiningOp<cir::CastOp>()) {
                val = cast.getSrc();
            }
            return val;
        };

        // Helper: Find the original stack allocation (cir.alloca) for a loaded variable
        auto getBackingAlloca = [&](Value v) -> Operation* {
            v = stripCasts(v);
            if (auto loadOp = v.getDefiningOp<cir::LoadOp>()) {
                return loadOp.getAddr().getDefiningOp();
            }
            return nullptr;
        };

        Operation* offsetAlloc = getBackingAlloca(offsetArg);
        if (!offsetAlloc || !isa<cir::AllocaOp>(offsetAlloc)) {
            llvm::errs() << "[IOOpt] ABORT: Offset is not a traceable local variable.\n";
            return failure();
        }
        Value offsetPtr = offsetAlloc->getResult(0);

        // Scan the loop body for EVERY time the offset variable is mutated
        SmallVector<cir::StoreOp, 2> offsetMutations;
        forOp.walk([&](cir::StoreOp store) {
            if (store.getAddr() == offsetPtr) {
                offsetMutations.push_back(store);
            }
        });

        // To be perfectly contiguous, it must advance exactly once per loop iteration
        if (offsetMutations.size() != 1) {
            llvm::errs() << "[IOOpt] ABORT: Offset is mutated multiple times or not at all (Random Access).\n";
            return failure();
        }

        cir::StoreOp offsetUpdate = offsetMutations.front();
        Value storedVal = stripCasts(offsetUpdate.getValue());

        // The mutation MUST be an addition
        auto addOp = storedVal.getDefiningOp<cir::BinOp>();
        if (!addOp || addOp.getKind() != cir::BinOpKind::Add) {
            llvm::errs() << "[IOOpt] ABORT: Offset mutation is not an addition.\n";
            return failure();
        }

        Value lhs = stripCasts(addOp.getLhs());
        Value rhs = stripCasts(addOp.getRhs());

        // Check if a value is a load of our offset pointer
        auto isLoadOfOffset = [&](Value v) {
            if (auto load = v.getDefiningOp<cir::LoadOp>()) {
                return load.getAddr() == offsetPtr;
            }
            return false;
        };

        // Check if a value perfectly matches the size argument of the I/O call
        auto isSizeEquivalence = [&](Value v) {
            return v == stripCasts(sizeArg);
        };

        // Prove: Offset_Next = Offset_Current + Size (or Size + Offset_Current)
        bool validAdvancement = (isLoadOfOffset(lhs) && isSizeEquivalence(rhs)) ||
                                (isLoadOfOffset(rhs) && isSizeEquivalence(lhs));

        if (!validAdvancement) {
            llvm::errs() << "[IOOpt] ABORT: Offset does not advance perfectly contiguously by 'size'.\n";
            return failure();
        }

        // We successfully proved contiguity
        // We must capture the value of the offset BEFORE the loop starts to pass to preadv/pwritev
        rewriter.setInsertionPoint(forOp);
        initialOffsetForSyscall = cir::LoadOp::create(
            rewriter,
            forOp.getLoc(), 
            mlir::cast<cir::PointerType>(offsetPtr.getType()).getPointee(), 
            offsetPtr
        );

        llvm::errs() << "[IOOpt] SUCCESS: Positional offset mathematically proven to be contiguous!\n";
    }

    // 2. Extract Upper Bound
    int64_t upperBound = -1;
    condRegion.walk([&](cir::CmpOp cmp) {
      if (auto constOp = cmp.getOperand(1).getDefiningOp<cir::ConstantOp>()) {
        auto attr = constOp.getValue();
        if (auto cirInt = dyn_cast<cir::IntAttr>(attr)) {
          upperBound = cirInt.getValue().getSExtValue();
        } else if (auto stdInt = dyn_cast<IntegerAttr>(attr)) {
          upperBound = stdInt.getInt();
        }
      }
    });

    if (upperBound <= 1) return failure();

    // 3. Extract Step Value
    int64_t stepValue = 1; 
    bool isValidStep = false; 

    Region &stepRegion = forOp.getStep();
    stepRegion.walk([&](cir::BinOp binOp) {
      if (binOp.getKind() != cir::BinOpKind::Add) {
        isValidStep = false;
        return WalkResult::interrupt(); 
      }
      if (auto constOp = binOp.getOperand(1).getDefiningOp<cir::ConstantOp>()) {
        auto attr = constOp.getValue();
        if (auto cirInt = mlir::dyn_cast<cir::IntAttr>(attr)) {
          stepValue = cirInt.getValue().getSExtValue();
          isValidStep = true;
        } else if (auto stdInt = mlir::dyn_cast<IntegerAttr>(attr)) {
          stepValue = stdInt.getInt();
          isValidStep = true;
        }
      }
      return WalkResult::advance(); 
    });

    if (!isValidStep) return failure();

    int64_t lowerBound = 0; 
    int64_t tripCount = (upperBound - lowerBound + stepValue - 1) / stepValue;

    // --- PHASE 1: PRE-LOOP ALLOCATIONS ---
    Location loc = forOp.getLoc();
    rewriter.setInsertionPoint(forOp);

    auto ctx = rewriter.getContext();
    Value tripCountVal = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)tripCount);

    auto stdI32Ty = rewriter.getI32Type();
    auto stdI64Ty = rewriter.getI64Type();
    
    auto ptrArrayType = MemRefType::get({ShapedType::kDynamic}, stdI64Ty);
    auto sizeArrayType = MemRefType::get({ShapedType::kDynamic}, stdI64Ty);

    Value ptrsMemref = memref::AllocaOp::create(rewriter, loc, ptrArrayType, ValueRange{tripCountVal});
    Value sizesMemref = memref::AllocaOp::create(rewriter, loc, sizeArrayType, ValueRange{tripCountVal});

    auto idxArrayType = MemRefType::get({1}, rewriter.getIndexType());
    Value idxAlloca = memref::AllocaOp::create(rewriter, loc, idxArrayType);
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)0);
    memref::StoreOp::create(rewriter, loc, zeroIdx, idxAlloca, ValueRange{zeroIdx});

    auto fdStashType = MemRefType::get({1}, stdI32Ty);
    Value fdStash = memref::AllocaOp::create(rewriter, loc, fdStashType);

    // --- PHASE 2: INSIDE THE LOOP ---
    rewriter.modifyOpInPlace(forOp, [&]() {
        rewriter.setInsertionPoint(ioCall);

        Value fdArg = ioCall.getOperand(0);
        Value bufArg = ioCall.getOperand(1);
        Value lenArg = ioCall.getOperand(2);

        Value stdFd = io::IOCastOp::create(rewriter, loc, stdI32Ty, fdArg);
        Value stdBuf = io::IOCastOp::create(rewriter, loc, stdI64Ty, bufArg);
        Value stdLen = io::IOCastOp::create(rewriter, loc, stdI64Ty, lenArg);

        memref::StoreOp::create(rewriter, loc, stdFd, fdStash, ValueRange{zeroIdx});

        Value currentIdx = memref::LoadOp::create(rewriter, loc, idxAlloca, ValueRange{zeroIdx});
        memref::StoreOp::create(rewriter, loc, stdBuf, ptrsMemref, ValueRange{currentIdx});
        memref::StoreOp::create(rewriter, loc, stdLen, sizesMemref, ValueRange{currentIdx});

        Value oneIdx = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)1);
        Value nextIdx = arith::AddIOp::create(rewriter, loc, currentIdx, oneIdx);
        memref::StoreOp::create(rewriter, loc, nextIdx, idxAlloca, ValueRange{zeroIdx});

        rewriter.eraseOp(ioCall);
    });

    // --- PHASE 3: POST-LOOP ---
    rewriter.setInsertionPointAfter(forOp);
    Value finalFd = memref::LoadOp::create(rewriter, loc, fdStash, ValueRange{zeroIdx});

    if (isRead) {
        io::BatchReadVOp::create(rewriter, loc, stdI64Ty, finalFd, ptrsMemref, sizesMemref, tripCountVal);
        llvm::errs() << "[IOOpt] AMAZING SUCCESS: Loop optimized to BatchReadVOp!\n";
    } else {
        io::BatchWriteVOp::create(rewriter, loc, stdI64Ty, finalFd, ptrsMemref, sizesMemref, tripCountVal);
        llvm::errs() << "[IOOpt] AMAZING SUCCESS: Loop optimized to BatchWriteVOp!\n";
    }

    return success();
  }
};

struct HoistWriteLoopPattern {
  static LogicalResult matchAndRewrite(scf::ForOp loop, IRRewriter &rewriter) {
    Block *body = loop.getBody();
    if (!body || body->empty()) return failure();

    io::WriteOp writeOp = nullptr;
    bool hasSideEffects = false;

    // Detect if this is a pure I/O loop
    for (Operation &op : *body) {
      if (isa<scf::YieldOp>(op)) continue; 

      if (auto ioWrite = dyn_cast<io::WriteOp>(op)) {
        if (writeOp) { hasSideEffects = true; break; }
        writeOp = ioWrite;
      } else if (!isMemoryEffectFree(&op)) {
        hasSideEffects = true; break; 
      }
    }

    if (hasSideEffects || !writeOp) return failure();
    if (!loop.isDefinedOutsideOfLoop(writeOp.getFd())) return failure();


    rewriter.setInsertionPoint(loop);

    Location loc = loop.getLoc();
    Value diff = arith::SubIOp::create(rewriter, loc, loop.getUpperBound(), loop.getLowerBound());
    Value tripCount = arith::DivSIOp::create(rewriter, loc, diff, loop.getStep());
    Value tripCountI64 = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), tripCount);

    // Safe Size Hoisting 
    Value safeSize = writeOp.getSize();
    if (!loop.isDefinedOutsideOfLoop(safeSize)) {
        Operation *defOp = safeSize.getDefiningOp();
        if (defOp && defOp->hasTrait<OpTrait::ConstantLike>()) {
            // Clone the constant outside the loop so it survives deletion
            safeSize = rewriter.clone(*defOp)->getResult(0);
        } else {
            return failure();
        }
    }

    // Contiguous vs Vector Routing
    Value basePointer;
    if (isContiguousMemoryAccess(writeOp.getBuffer(), loop, safeSize, basePointer)) {
        // Contiguous writes (Use safeSize!)
        Value totalSize = arith::MulIOp::create(rewriter, loc, tripCountI64, safeSize);
        io::BatchWriteOp::create(rewriter, loc, rewriter.getI64Type(), writeOp.getFd(), basePointer, totalSize);
    } else {
        // Fallback to scattered writes (writev) 
        auto ptrArrayType = MemRefType::get({ShapedType::kDynamic}, writeOp.getBuffer().getType());
        auto sizeArrayType = MemRefType::get({ShapedType::kDynamic}, rewriter.getI64Type());
        
        Value ptrsMemref = memref::AllocaOp::create(rewriter, loc, ptrArrayType, tripCount);
        Value sizesMemref = memref::AllocaOp::create(rewriter, loc, sizeArrayType, tripCount);

        auto calcLoop = scf::ForOp::create(rewriter, loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep());
        rewriter.setInsertionPointToStart(calcLoop.getBody());

        Value currentIV = calcLoop.getInductionVar();
        Value ivOffset = arith::SubIOp::create(rewriter, loc, currentIV, loop.getLowerBound());
        Value arrayIdx = arith::DivSIOp::create(rewriter, loc, ivOffset, loop.getStep());
        
        IRMapping mapping;
        mapping.map(loop.getInductionVar(), currentIV);
        for (Operation &op : *body) {
            if (isa<io::WriteOp, scf::YieldOp>(op)) continue;
            rewriter.clone(op, mapping);
        }

        Value mappedBuffer = mapping.lookupOrDefault(writeOp.getBuffer());
        Value mappedSize = mapping.lookupOrDefault(writeOp.getSize());
        
        // Use arrayIdx directly
        memref::StoreOp::create(rewriter, loc, mappedBuffer, ptrsMemref, arrayIdx);
        memref::StoreOp::create(rewriter, loc, mappedSize, sizesMemref, arrayIdx);

        rewriter.setInsertionPointAfter(calcLoop);
        io::BatchWriteVOp::create(rewriter, loc, rewriter.getI64Type(), writeOp.getFd(), ptrsMemref, sizesMemref, tripCount);
    }

    // Completely erase the original write loop
    rewriter.eraseOp(loop);

    return success();
  }
};

struct HoistReadLoopPattern {
  static LogicalResult matchAndRewrite(scf::ForOp loop, IRRewriter &rewriter) {
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

    // This guarantees the analysis is perfectly synchronized with the current IR state.
    AliasAnalysis aliasAnalysis(loop);

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
            // Ask our fresh Alias Analysis if the memory overlaps
            AliasResult aliasResult = aliasAnalysis.alias(readOp.getBuffer(), writtenVal);
            if (!aliasResult.isNo()) {
              return failure(); 
            }
          } else {
            return failure();
          }
        }
      } else if (!isMemoryEffectFree(&op)) {
         return failure();
      }
    }

    rewriter.setInsertionPoint(loop);

    Location loc = loop.getLoc();
    Value diff = arith::SubIOp::create(rewriter, loc, loop.getUpperBound(), loop.getLowerBound());
    Value tripCount = arith::DivSIOp::create(rewriter, loc, diff, loop.getStep());
    Value tripCountI64 = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), tripCount);

    Value safeSize = readOp.getSize();
    if (!loop.isDefinedOutsideOfLoop(safeSize)) {
        Operation *defOp = safeSize.getDefiningOp();
        if (defOp && defOp->hasTrait<OpTrait::ConstantLike>()) {
            safeSize = rewriter.clone(*defOp)->getResult(0);
        } else {
            return failure();
        }
    }

    // Contiguous vs Vector Routing
    Value basePointer;
    if (isContiguousMemoryAccess(readOp.getBuffer(), loop, safeSize, basePointer)) {
        Value totalSize = arith::MulIOp::create(rewriter, loc, tripCountI64, safeSize);
        io::BatchReadOp::create(rewriter, loc, rewriter.getI64Type(), readOp.getFd(), basePointer, totalSize);
    } else {
        auto ptrArrayType = MemRefType::get({ShapedType::kDynamic}, readOp.getBuffer().getType());
        auto sizeArrayType = MemRefType::get({ShapedType::kDynamic}, rewriter.getI64Type());

        Value ptrsMemref = memref::AllocaOp::create(rewriter, loc, ptrArrayType, tripCount);
        Value sizesMemref = memref::AllocaOp::create(rewriter, loc, sizeArrayType, tripCount);

        auto calcLoop = scf::ForOp::create(rewriter, loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep());
        rewriter.setInsertionPointToStart(calcLoop.getBody());

        Value currentIV = calcLoop.getInductionVar();
        Value ivOffset = arith::SubIOp::create(rewriter, loc, currentIV, loop.getLowerBound());
        Value arrayIdx = arith::DivSIOp::create(rewriter, loc, ivOffset, loop.getStep());

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
        
        // Use arrayIdx directly
        memref::StoreOp::create(rewriter, loc, mappedBuffer, ptrsMemref, arrayIdx);
        memref::StoreOp::create(rewriter, loc, mappedSize, sizesMemref, arrayIdx);

        rewriter.setInsertionPoint(loop); 
        io::BatchReadVOp::create(rewriter, loc, rewriter.getI64Type(), readOp.getFd(), ptrsMemref, sizesMemref, tripCount);
    }

    rewriter.modifyOpInPlace(loop, [&]() {
        rewriter.setInsertionPoint(readOp);
        rewriter.replaceOp(readOp, safeSize);
    });

    return success();
  }
};

struct StripCIRAttrsPass : public PassWrapper<StripCIRAttrsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StripCIRAttrsPass)

  llvm::StringRef getArgument() const final { return "strip-cir-attrs"; }
  llvm::StringRef getDescription() const final { return "Strips leftover CIR attributes before LLVM translation."; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
   
    mlir::io::bootstrapTargetInfo(module);
 
    // Walk every single operation in the entire file (Functions, Globals, Blocks)
    module.walk([&](Operation *op) {
      SmallVector<StringAttr, 4> attrsToRemove;
      
      // Look at all attributes attached to this operation
      for (NamedAttribute attr : op->getAttrs()) {
        llvm::StringRef name = attr.getName().getValue();
        
        // If it's a ClangIR leftover, mark it for death
        if (name.starts_with("cir.") || 
            name == "cxx_special_member" || 
            name == "global_visibility" ||
            name == "sym_visibility") {
          attrsToRemove.push_back(attr.getName());
        }
      }
      
      // Remove them cleanly using the MLIR API
      for (StringAttr attrName : attrsToRemove) {
        op->removeAttr(attrName);
      }
    });
  }
};

struct IOLoopBatchingPass : public PassWrapper<IOLoopBatchingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IOLoopBatchingPass)

  llvm::StringRef getArgument() const final { return "io-loop-batching"; }
  llvm::StringRef getDescription() const final { return "Batches I/O operations inside loops."; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<io::IODialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    llvm::errs() << "\n[IOOpt] ---> Pass activated on Module! <__-\n";

    ModuleOp module = getOperation();
    mlir::io::bootstrapTargetInfo(module);
    MLIRContext *context = &getContext();
    IRRewriter rewriter(context);

    // Safely collect all loops first to avoid iterator invalidation
    SmallVector<cir::ForOp> cirLoops;
    SmallVector<scf::ForOp> scfLoops;

    module.walk([&](Operation *op) {
      if (auto cirLoop = dyn_cast<cir::ForOp>(op)) {
        cirLoops.push_back(cirLoop);
      } else if (auto scfLoop = dyn_cast<scf::ForOp>(op)) {
        scfLoops.push_back(scfLoop);
      }
    });

    // Deterministically apply the CIR pattern
   for (cir::ForOp loop : cirLoops) {
      // Call the static function directly
      (void)CirLoopBatchingPattern::matchAndRewrite(loop, rewriter);
    }

    // Deterministically apply the SCF patterns
    for (scf::ForOp loop : scfLoops) {
      if (succeeded(HoistWriteLoopPattern::matchAndRewrite(loop, rewriter))) {
        continue; 
      }
      (void)HoistReadLoopPattern::matchAndRewrite(loop, rewriter);
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

std::unique_ptr<mlir::Pass> createStripCIRAttrsPass() {
  return std::make_unique<StripCIRAttrsPass>(); 
}

// Register the pass so `io-opt` knows it exists
void registerIOPasses() {
  mlir::PassRegistration<IOLoopBatchingPass>();
  mlir::PassRegistration<StripCIRAttrsPass>();
}

} // namespace io
} // namespace mlir
