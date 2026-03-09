#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Support/CommandLine.h"

#include <vector>
#include <cstdlib>
#include <string>

using namespace llvm;


// Helper function to read from the environment at compile-time
static unsigned getEnvOrDefault(const char* Name, unsigned Default) {
  if (const char* Val = std::getenv(Name)) return std::stoi(Val);
  return Default;
}

// Global tuning parameters
static unsigned IOBatchThreshold = 4;
static unsigned IOShadowBufferSize = 4096;
static unsigned IOHighWaterMark = 65536;

namespace {

  struct IOArgs {
    Value *Target; 
    Value *Buffer; 
    Value *Length; 
    enum { NONE, C_FWRITE, C_FREAD, POSIX_WRITE, POSIX_READ, POSIX_PWRITE, POSIX_PREAD, CXX_WRITE, CXX_READ, MPI_WRITE_AT, MPI_READ_AT } Type;
  };

  IOArgs getIOArguments(CallInst *Call) {
    Function *F = Call->getCalledFunction();
    if (!F || !F->hasName() || !F->isDeclaration()) return {nullptr, nullptr, nullptr, IOArgs::NONE};

    std::string Demangled = llvm::demangle(F->getName().str());
    
    // Explicit-offset POSIX calls
    if (Demangled == "pread" || Demangled == "pread64") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PREAD};
    if (Demangled == "pwrite" || Demangled == "pwrite64") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PWRITE};

    // Implicit-offset POSIX calls
    if (Demangled == "write" || Demangled == "write64") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_WRITE};
    if (Demangled == "read" || Demangled == "read64")   return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_READ};

    // Buffered C-Library
    if (Demangled == "fwrite") return {Call->getArgOperand(3), Call->getArgOperand(0), Call->getArgOperand(2), IOArgs::C_FWRITE};
    if (Demangled == "fread")  return {Call->getArgOperand(3), Call->getArgOperand(0), Call->getArgOperand(2), IOArgs::C_FREAD};

    // Signature: (fh, offset, buf, count, datatype, status)
    if (Demangled == "MPI_File_write_at" || Demangled == "PMPI_File_write_at") return {Call->getArgOperand(0), Call->getArgOperand(2), Call->getArgOperand(3), IOArgs::MPI_WRITE_AT};
    if (Demangled == "MPI_File_read_at" || Demangled == "PMPI_File_read_at")  return {Call->getArgOperand(0), Call->getArgOperand(2), Call->getArgOperand(3), IOArgs::MPI_READ_AT};

    // C++ Streams
    if ((Demangled.find("std::basic_ostream") != std::string::npos || 
         Demangled.find("std::ostream") != std::string::npos) && Demangled.find("::write") != std::string::npos) {
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::CXX_WRITE};
    }
    
    return {nullptr, nullptr, nullptr, IOArgs::NONE};
  }

  bool checkAdjacency(Value *BufA, Value *LenA, Value *BufB, const DataLayout &DL, ScalarEvolution *SE = nullptr) {
    if (SE) {
      const SCEV *A = SE->getSCEV(BufA);
      const SCEV *Len = SE->getSCEV(LenA);
      const SCEV *B = SE->getSCEV(BufB);

      if (!isa<SCEVCouldNotCompute>(A) && !isa<SCEVCouldNotCompute>(B) && !isa<SCEVCouldNotCompute>(Len)) {
	// Handle bit-width differences (i32 vs i64)
	// We extend the length to match the pointer/offset width to avoid APInt crashes
	const SCEV *ExtendedLen = SE->getTruncateOrZeroExtend(Len, A->getType());
	const SCEV *ExpectedB = SE->getAddExpr(A, ExtendedLen);
	if (ExpectedB == B) return true;
      }
    }

    // Fallback: Pointer-base striping (Handles simple C-style array math)
    auto *BaseA = BufA->stripPointerCasts();
    auto *BaseB = BufB->stripPointerCasts();
    if (auto *GEP = dyn_cast<GEPOperator>(BaseB)) {
      if (GEP->getPointerOperand() == BaseA) {
	APInt Offset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
	if (GEP->accumulateConstantOffset(DL, Offset)) {
	  if (auto *LenConst = dyn_cast<ConstantInt>(LenA)) {
	    // Use getZExtValue to safely compare across different bit widths
	    return Offset.getZExtValue() == LenConst->getZExtValue();
	  }
	}
      }
    }
    return false;
  }

  bool isSafeToAddToBatch(const std::vector<CallInst*> &Batch, CallInst *NewCall, AAResults &AA, const DataLayout &DL, ScalarEvolution &SE) {
    if (Batch.empty()) return true;

    CallInst *LastCall = Batch.back();
    if (LastCall->getCalledFunction() != NewCall->getCalledFunction()) return false;

    IOArgs FirstArgs = getIOArguments(Batch.front());
    IOArgs LastArgs = getIOArguments(LastCall);
    IOArgs NewArgs = getIOArguments(NewCall);

    // Disk Offset Contiguity Tracking ---
    if (NewArgs.Type == IOArgs::POSIX_PREAD || NewArgs.Type == IOArgs::POSIX_PWRITE) {
      Value *LastOffset = LastCall->getArgOperand(3);
      Value *NewOffset = NewCall->getArgOperand(3);
      Value *LastLen = LastArgs.Length;
      bool isContiguous = false;
        
      if (SE.isSCEVable(LastOffset->getType()) && SE.isSCEVable(NewOffset->getType()) && SE.isSCEVable(LastLen->getType())) {
	const SCEV *SLast = SE.getSCEV(LastOffset);
	const SCEV *SNew = SE.getSCEV(NewOffset);
	const SCEV *SLen = SE.getSCEV(LastLen);
	if (!isa<SCEVCouldNotCompute>(SLast) && !isa<SCEVCouldNotCompute>(SNew) && !isa<SCEVCouldNotCompute>(SLen)) {
	  if (SLast->getType() == SLen->getType()) {
	    const SCEV *ExpectedNext = SE.getAddExpr(SLast, SLen);
	    if (ExpectedNext == SNew) isContiguous = true;
	  }
	}
      }
      if (!isContiguous) {
	if (auto *CLastOff = dyn_cast<ConstantInt>(LastOffset)) {
	  if (auto *CNewOff = dyn_cast<ConstantInt>(NewOffset)) {
	    if (auto *CLen = dyn_cast<ConstantInt>(LastLen)) {
	      if (CLastOff->getZExtValue() + CLen->getZExtValue() == CNewOff->getZExtValue()) isContiguous = true;
	    }
	  }
	}
      }
      if (!isContiguous) return false;

    } else if (NewArgs.Type == IOArgs::MPI_READ_AT || NewArgs.Type == IOArgs::MPI_WRITE_AT) {
      // MPI Smart Datatype Checking (Resolves OpenMPI opaque pointer loads)
      Value *DT1 = LastCall->getArgOperand(4);
      Value *DT2 = NewCall->getArgOperand(4);
      if (DT1 != DT2) {
	auto *L1 = dyn_cast<LoadInst>(DT1);
	auto *L2 = dyn_cast<LoadInst>(DT2);
	if (!L1 || !L2 || L1->getPointerOperand() != L2->getPointerOperand()) return false;
      }
      // MPI Strict Datatype Checking
      if (LastCall->getArgOperand(4) != NewCall->getArgOperand(4)) return false; 

      Value *LastOffset = LastCall->getArgOperand(1);
      Value *NewOffset = NewCall->getArgOperand(1);
      Value *LastCount = LastArgs.Length; 
        
      bool isContiguous = false;
      if (SE.isSCEVable(LastOffset->getType()) && SE.isSCEVable(NewOffset->getType()) && SE.isSCEVable(LastCount->getType())) {
	const SCEV *SLast = SE.getSCEV(LastOffset);
	const SCEV *SNew = SE.getSCEV(NewOffset);
	const SCEV *SCount = SE.getSCEV(LastCount);
	if (!isa<SCEVCouldNotCompute>(SLast) && !isa<SCEVCouldNotCompute>(SNew) && !isa<SCEVCouldNotCompute>(SCount)) {
	  // Safely extend the count to match the offset bit-width
	  const SCEV *ExtendedCount = SCount;
	  if (SCount->getType()->getIntegerBitWidth() < SLast->getType()->getIntegerBitWidth()) {
	    ExtendedCount = SE.getZeroExtendExpr(SCount, SLast->getType());
	  }
	  const SCEV *ExpectedNext = SE.getAddExpr(SLast, ExtendedCount);
	  if (ExpectedNext == SNew) isContiguous = true;
	} 
      }
        
      // Fallback: If SCEV fails, see if they are raw Constants we can add manually
      if (!isContiguous) {
	if (auto *CLastOff = dyn_cast<ConstantInt>(LastOffset)) {
	  if (auto *CNewOff = dyn_cast<ConstantInt>(NewOffset)) {
	    if (auto *CLen = dyn_cast<ConstantInt>(LastCount)) {
	      if (CLastOff->getZExtValue() + CLen->getZExtValue() == CNewOff->getZExtValue()) {
		isContiguous = true;
	      }
	    }
	  }
	}
      }

      if (!isContiguous) return false;

    } else if (NewArgs.Type == IOArgs::POSIX_READ || NewArgs.Type == IOArgs::POSIX_WRITE) {
      // Implicit offsets natively guarantee contiguity if File Descriptors match!
      if (NewArgs.Target != LastArgs.Target) return false; 
    }

    // Target Matching Logic
    bool TargetsMatch = (FirstArgs.Target == NewArgs.Target);
    LoadInst *Load1 = dyn_cast<LoadInst>(FirstArgs.Target);
    LoadInst *Load2 = dyn_cast<LoadInst>(NewArgs.Target);
    
    if (!TargetsMatch) {
      if (Load1 && Load2 && Load1->getPointerOperand() == Load2->getPointerOperand()) {
        TargetsMatch = true;
      }
    }
    if (!TargetsMatch) return false;

    // Helper for precise sizes
    auto getPreciseLoc = [&](Value *Buf, Value *Len) {
      if (auto *C = dyn_cast<ConstantInt>(Len))
        return MemoryLocation(Buf, LocationSize::precise(C->getZExtValue()));
      return MemoryLocation(Buf, LocationSize::beforeOrAfterPointer());
    };

    MemoryLocation NewLoc = getPreciseLoc(NewArgs.Buffer, NewArgs.Length);

    Instruction *I = LastCall->getNextNode();
    BasicBlock *CurrBB = LastCall->getParent();

    while (I != NewCall) {
      // If we hit the end of a Basic Block, move to the successor
      if (!I) {
	Instruction *Term = CurrBB->getTerminator();         
   
	// Safety: We only follow "Success" paths. If there are multiple 
	// successors (like a switch), or if it's a return, we must abort.
	BasicBlock *NextBB = nullptr;
	if (auto *BI = dyn_cast<BranchInst>(Term)) {
	  if (BI->isConditional()) {
	    // Follow the path that DOES NOT return (the success path)
	    for (BasicBlock *Succ : BI->successors()) {
	      if (!isa<ReturnInst>(Succ->getTerminator())) {
		NextBB = Succ;
		break;
	      }
	    }
	  } else {
	    NextBB = BI->getSuccessor(0);
	  }
	}

	// Only proceed if the path is linear (Single Predecessor)
	if (!NextBB || NextBB->getSinglePredecessor() != CurrBB) return false;

	CurrBB = NextBB;
	I = &CurrBB->front();
	continue; 
      }

      // --- Existing Analysis Logic (Keep this part) ---
      if (auto *CI = dyn_cast<CallInst>(I)) {
	if (CI->getIntrinsicID() == Intrinsic::lifetime_end ||
	    CI->getIntrinsicID() == Intrinsic::lifetime_start) return false;
            
	if (getIOArguments(CI).Type != IOArgs::NONE) return false;
	if (!CI->onlyReadsMemory() && !CI->doesNotAccessMemory()) return false;
      }

      if (I->mayReadOrWriteMemory()) {
	// Check if this instruction mutates the new buffer
	if (isModSet(AA.getModRefInfo(I, NewLoc))) return false;

	// Check if this instruction mutates any previously batched buffers
	for (CallInst *BatchedCall : Batch) {
	  IOArgs BArgs = getIOArguments(BatchedCall);
	  MemoryLocation BLoc = getPreciseLoc(BArgs.Buffer, BArgs.Length);
	  if (isModSet(AA.getModRefInfo(I, BLoc))) return false;
	}

	// Check for File Descriptor / Target mutations
	if (FirstArgs.Target->getType()->isPointerTy()) {
	  MemoryLocation TargetLoc(FirstArgs.Target, LocationSize::beforeOrAfterPointer());
	  if (isModSet(AA.getModRefInfo(I, TargetLoc))) return false;
	}
      }
      // --- End of Analysis Logic ---

      I = I->getNextNode();
    }

    return true;
  }

  // ====================================================================
  // Define the I/O Patterns
  // ====================================================================
  enum class IOPattern {
    Contiguous,   // Perfectly sequential memory (array[0], array[1])
    Strided,      // Uniform scattered sizes perfect for SIMD Gather
    ShadowBuffer, // Small scattered writes we can safely pack on the stack
    Vectored,     // Heavy scattered I/O (Requires readv/writev arrays)
    Unprofitable  // Too few calls or lacks byte-weight. Do not optimise
  };

  // ====================================================================
  // Classifier: Analyes the batch and decides the best strategy
  // ====================================================================
  IOPattern classifyBatch(const std::vector<CallInst*> &Batch, const DataLayout &DL, uint64_t &OutTotalConstSize, ScalarEvolution *SE = nullptr) {
    if (Batch.size() <= 1) return IOPattern::Unprofitable;

    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isRead = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD || FirstArgs.Type == IOArgs::POSIX_PREAD);

    bool AllContiguous = true;
    for (size_t i = 0; i < Batch.size() - 1; ++i) {
      IOArgs A = getIOArguments(Batch[i]);
      IOArgs B = getIOArguments(Batch[i+1]);
      if (!checkAdjacency(A.Buffer, A.Length, B.Buffer, DL, SE)) {
        AllContiguous = false;
        break;
      }
    }
    if (AllContiguous) return IOPattern::Contiguous;

    // --- SCATTERED I/O ROUTING ---
    // If we reach here, the batch has memory or disk gaps.

    // We do not currently have Vectored or ShadowBuffer code generators for MPI-IO.
    // If MPI calls are not perfectly contiguous, we must safely bail out
    if (FirstArgs.Type == IOArgs::MPI_READ_AT || FirstArgs.Type == IOArgs::MPI_WRITE_AT) {
      return IOPattern::Unprofitable;
    }

    // If it has gaps, it muar be raw POSIX to use writev/preadv.
    // (This safely rejects C-Library fwrite and C++ streams, which don't support writev)
    if (FirstArgs.Type != IOArgs::POSIX_WRITE && FirstArgs.Type != IOArgs::POSIX_READ &&
        FirstArgs.Type != IOArgs::POSIX_PWRITE && FirstArgs.Type != IOArgs::POSIX_PREAD) {
      return IOPattern::Unprofitable;
    }

    bool isStrided = false;
    uint64_t UniformSize = 0;
    
    if (!isRead && Batch.size() > 1) {
      if (auto *FirstSizeC = dyn_cast<ConstantInt>(getIOArguments(Batch.front()).Length)) {
	UniformSize = FirstSizeC->getZExtValue();
	if (UniformSize == 1 || UniformSize == 2 || UniformSize == 4 || UniformSize == 8) {
	  isStrided = true;
	  for (CallInst *C : Batch) {
	    auto *SizeC = dyn_cast<ConstantInt>(getIOArguments(C).Length);
	    if (!SizeC || SizeC->getZExtValue() != UniformSize) {
	      isStrided = false;
	      break;
	    }
	  }
	}
      }
    }

    if (isStrided) {
      OutTotalConstSize = UniformSize; 
      if (Batch.size() >= 4 && Batch.size() <= 64) {
	return IOPattern::Strided;
      }
    }

    size_t DynamicThreshold = IOBatchThreshold; 
    uint64_t TotalConstSize = 0;
    bool AllSizesConstant = true;

    for (CallInst *C : Batch) {
      if (auto *CI = dyn_cast<ConstantInt>(getIOArguments(C).Length)) {
        TotalConstSize += CI->getZExtValue();
      } else {
        AllSizesConstant = false;
      }
    }
    
    OutTotalConstSize = TotalConstSize;

    if (isRead) {
      DynamicThreshold = 2; 
    } else {
      Function *F = Batch.back()->getFunction();
      if (F->getInstructionCount() > 150) DynamicThreshold = 3;
      if (AllSizesConstant && TotalConstSize > 128) DynamicThreshold = 3;
    }

    if (Batch.size() >= DynamicThreshold) {
      return IOPattern::Vectored;
    }

    if (!isRead && AllSizesConstant && TotalConstSize > 0 && TotalConstSize <= IOShadowBufferSize) {
      return IOPattern::ShadowBuffer;
    }

    return IOPattern::Unprofitable;
  }

  // ====================================================================
  // 3. The Router: Executes the IR transformations based on the Classifier
  // ====================================================================
  bool flushBatch(std::vector<CallInst*> &Batch, Module *M, ScalarEvolution &SE) {
    if (Batch.empty()) return false;

    const DataLayout &DL = M->getDataLayout();
    uint64_t TotalConstSize = 0;
    
    IOPattern Pattern = classifyBatch(Batch, DL, TotalConstSize, &SE);

    if (Pattern == IOPattern::Unprofitable) {
      Batch.clear();
      return false; 
    }

    IRBuilder<> Builder(Batch.back());
    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isRead = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD || FirstArgs.Type == IOArgs::POSIX_PREAD);
    bool isExplicit = (FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::POSIX_PWRITE);

    switch (Pattern) {
    case IOPattern::Contiguous: {
      Value *TotalLen = FirstArgs.Length;
      for (size_t i = 1; i < Batch.size(); ++i) {
	TotalLen = Builder.CreateAdd(TotalLen, getIOArguments(Batch[i]).Length, "sum.len");
      }
          
      std::vector<Value *> NewArgs;
      if (FirstArgs.Type == IOArgs::MPI_WRITE_AT || FirstArgs.Type == IOArgs::MPI_READ_AT) {
	// MPI Signature Reconstruction
	NewArgs = {
	  Batch[0]->getArgOperand(0), // fh
	  Batch[0]->getArgOperand(1), // offset
	  FirstArgs.Buffer,           // starting buf
	  TotalLen,                   // summed count
	  Batch[0]->getArgOperand(4), // datatype
	  Batch[0]->getArgOperand(5)  // status
	};
      } else if (FirstArgs.Type == IOArgs::C_FWRITE || FirstArgs.Type == IOArgs::C_FREAD) {
	NewArgs = {FirstArgs.Buffer, Batch[0]->getArgOperand(1), TotalLen, FirstArgs.Target};
      } else if (isExplicit) {
	NewArgs = {FirstArgs.Target, FirstArgs.Buffer, TotalLen, Batch[0]->getArgOperand(3)};
      } else {
	NewArgs = {FirstArgs.Target, FirstArgs.Buffer, TotalLen};
      }
      Builder.CreateCall(Batch[0]->getCalledFunction(), NewArgs);

      // Some applications expects each individual call to return the bytes written.
      // We loop through the batch and replace the old return values with constants.
      for (CallInst *C : Batch) {
	if (!C->use_empty()) {
	  // If the original code was checking 'if (bytes_written == expected)'
	  // we give it the 'expected' value so the check passes.
	  IOArgs OriginalArgs = getIOArguments(C);
	  C->replaceAllUsesWith(OriginalArgs.Length);
	}
      }

      errs() << "[IOOpt] SUCCESS: N-Way merged " << Batch.size() << " contiguous " << (isRead ? "reads" : "writes") << "!\n";
      break;
    }

    case IOPattern::Strided: {
      unsigned ElementBytes = TotalConstSize;
      unsigned NumElements = Batch.size();
        
      Type *ElementTy = Builder.getIntNTy(ElementBytes * 8); 
      auto *VecTy = FixedVectorType::get(ElementTy, NumElements);
      Value *GatherVec = PoisonValue::get(VecTy);
        
      for (unsigned i = 0; i < NumElements; ++i) {
	IOArgs Args = getIOArguments(Batch[i]);
	LoadInst *LoadedVal = Builder.CreateLoad(ElementTy, Args.Buffer, "strided.load");
	GatherVec = Builder.CreateInsertElement(GatherVec, LoadedVal, Builder.getInt32(i), "gather.insert");
      }
        
      Function *F = Batch.back()->getFunction();
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());
      AllocaInst *ContiguousBuf = EntryBuilder.CreateAlloca(VecTy, nullptr, "simd.shadow.buf");
      Builder.CreateStore(GatherVec, ContiguousBuf);
        
      Value *TotalLenVal = Builder.getIntN(FirstArgs.Length->getType()->getIntegerBitWidth(), ElementBytes * NumElements);
      Value *BufCast = Builder.CreatePointerCast(ContiguousBuf, Builder.getPtrTy());
        
      std::vector<Value *> NewArgs;
      if (FirstArgs.Type == IOArgs::C_FWRITE) {
	NewArgs = {BufCast, Batch[0]->getArgOperand(1), TotalLenVal, FirstArgs.Target};
      } else if (isExplicit) {
	NewArgs = {FirstArgs.Target, BufCast, TotalLenVal, Batch[0]->getArgOperand(3)};
      } else {
	NewArgs = {FirstArgs.Target, BufCast, TotalLenVal};
      }
      Builder.CreateCall(Batch[0]->getCalledFunction(), NewArgs);

      // Some applications expects each individual call to return the bytes written.
      // We loop through the batch and replace the old return values with constants.
      for (CallInst *C : Batch) {
	if (!C->use_empty()) {
	  // If the original code was checking 'if (bytes_written == expected)'
	  // we give it the 'expected' value so the check passes.
	  IOArgs OriginalArgs = getIOArguments(C);
	  C->replaceAllUsesWith(OriginalArgs.Length);
	}
      }
        
      errs() << "[IOOpt] SUCCESS: SIMD Gathered " << NumElements << " strided writes into 1!\n";
      break;
    }

    case IOPattern::ShadowBuffer: {
      Function *F = Batch.back()->getFunction();
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());
        
      Type *Int8Ty = Builder.getInt8Ty();
      ArrayType *ShadowArrTy = ArrayType::get(Int8Ty, TotalConstSize);
      AllocaInst *ShadowBuf = EntryBuilder.CreateAlloca(ShadowArrTy, nullptr, "shadow.buf");
      ShadowBuf->setAlignment(Align(4096)); // Force 4KB alignment for O_DIRECT device support
 
      uint64_t Offset = 0;
      for (size_t i = 0; i < Batch.size(); ++i) {
	IOArgs Args = getIOArguments(Batch[i]);
	uint64_t Len = cast<ConstantInt>(Args.Length)->getZExtValue();
	Value *DestPtr = Builder.CreateInBoundsGEP(ShadowArrTy, ShadowBuf, {Builder.getInt32(0), Builder.getInt32(Offset)});
	Builder.CreateMemCpy(DestPtr, Align(1), Args.Buffer, Align(1), Len);
	Offset += Len;
      }
        
      Value *TotalLenVal = Builder.getIntN(FirstArgs.Length->getType()->getIntegerBitWidth(), TotalConstSize);
      std::vector<Value *> NewArgs;
      if (FirstArgs.Type == IOArgs::C_FWRITE) {
	NewArgs = {ShadowBuf, Batch[0]->getArgOperand(1), TotalLenVal, FirstArgs.Target};
      } else if (isExplicit) {
	NewArgs = {FirstArgs.Target, ShadowBuf, TotalLenVal, Batch[0]->getArgOperand(3)};
      } else {
	NewArgs = {FirstArgs.Target, ShadowBuf, TotalLenVal};
      }
      Builder.CreateCall(Batch[0]->getCalledFunction(), NewArgs);

      // Some applications expects each individual call to return the bytes written.
      // We loop through the batch and replace the old return values with constants.
      for (CallInst *C : Batch) {
	if (!C->use_empty()) {
	  // If the original code was checking 'if (bytes_written == expected)'
	  // we give it the 'expected' value so the check passes.
	  IOArgs OriginalArgs = getIOArguments(C);
	  C->replaceAllUsesWith(OriginalArgs.Length);
	}
      }

      errs() << "[IOOpt] SUCCESS: Shadow Buffered " << Batch.size() << " writes into 1 (" << TotalConstSize << " bytes)!\n";
      break;
    }

    case IOPattern::Vectored: {
      Type *Int32Ty = Builder.getInt32Ty();
      Type *PtrTy = PointerType::getUnqual(M->getContext());
      Type *SizeTy = DL.getIntPtrType(M->getContext());
        
      StringRef FuncName;
      if (isExplicit) FuncName = isRead ? "preadv" : "pwritev";
      else FuncName = isRead ? "readv" : "writev";

      FunctionType *VecTy;
      if (isExplicit) {
	Type *OffsetTy = Batch[0]->getArgOperand(3)->getType(); 
	VecTy = FunctionType::get(SizeTy, {Int32Ty, PtrTy, Int32Ty, OffsetTy}, false);
      } else {
	VecTy = FunctionType::get(SizeTy, {Int32Ty, PtrTy, Int32Ty}, false);
      }
        
      FunctionCallee VecFunc = M->getOrInsertFunction(FuncName, VecTy);
        
      StructType *IovecTy = StructType::get(M->getContext(), {PtrTy, SizeTy});
      ArrayType *IovArrayTy = ArrayType::get(IovecTy, Batch.size());
        
      Function *F = Batch.back()->getFunction();
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());
      AllocaInst *IovArray = EntryBuilder.CreateAlloca(IovArrayTy, nullptr, "iovec.array.N");
        
      for (size_t i = 0; i < Batch.size(); ++i) {
	IOArgs Args = getIOArguments(Batch[i]);
	Value *IovPtr = Builder.CreateInBoundsGEP(IovArrayTy, IovArray, {Builder.getInt32(0), Builder.getInt32(i)});
	Builder.CreateStore(Args.Buffer, Builder.CreateStructGEP(IovecTy, IovPtr, 0));
	Builder.CreateStore(Builder.CreateIntCast(Args.Length, SizeTy, false), Builder.CreateStructGEP(IovecTy, IovPtr, 1));
      }
        
      Value *Fd = Builder.CreateIntCast(FirstArgs.Target, Int32Ty, false);
        
      if (isExplicit) {
	Value *StartOffset = Batch[0]->getArgOperand(3);
	Builder.CreateCall(VecFunc, {Fd, IovArray, Builder.getInt32(Batch.size()), StartOffset});
      } else {
	Builder.CreateCall(VecFunc, {Fd, IovArray, Builder.getInt32(Batch.size())});
      }
       
      // Some application expects each individual call to return the bytes written.
      // We loop through the batch and replace the old return values with constants.
      for (CallInst *C : Batch) {
	if (!C->use_empty()) {
	  // If the original code was checking 'if (bytes_written == expected)'
	  // we give it the 'expected' value so the check passes.
	  IOArgs OriginalArgs = getIOArguments(C);
	  C->replaceAllUsesWith(OriginalArgs.Length);
	}
      }
  
      errs() << "[IOOpt] SUCCESS: N-Way converted " << Batch.size() << " " 
	     << (isRead ? "reads" : "writes") << " to " << FuncName << "!\n";
      break;
    }

    case IOPattern::Unprofitable:
    default:
      break;
    }
        
    for (CallInst *C : Batch) C->eraseFromParent();
    Batch.clear();
    return true;
  }

  bool hoistRead(CallInst *ReadCall, AAResults &AA, const DataLayout &DL) {
    IOArgs Args = getIOArguments(ReadCall);
    if (!Args.Buffer) return false;
    MemoryLocation DestLoc(Args.Buffer, LocationSize::beforeOrAfterPointer());
    Instruction *InsertPoint = ReadCall;
    Instruction *CurrentInst = ReadCall->getPrevNode();
    BasicBlock *CurrentBB = ReadCall->getParent();
    
    while (true) {
      if (CurrentInst) {
        if (!CurrentInst->isTerminator() && !isa<PHINode>(CurrentInst)) {
          bool DependsOnPrev = false;
          for (Value *Op : ReadCall->operands()) if (Op == CurrentInst) { DependsOnPrev = true; break; }
          if (DependsOnPrev) break;
          if (CurrentInst->mayReadOrWriteMemory()) {
            if (isModOrRefSet(AA.getModRefInfo(CurrentInst, DestLoc))) break;
            MemoryLocation TargetLoc(Args.Target, LocationSize::beforeOrAfterPointer());
            if (isModSet(AA.getModRefInfo(CurrentInst, TargetLoc))) break;
          }
          InsertPoint = CurrentInst;
        }
        CurrentInst = CurrentInst->getPrevNode();
      } else {
        BasicBlock *PredBB = CurrentBB->getSinglePredecessor();
        if (!PredBB || PredBB->getTerminator()->getNumSuccessors() > 1) break;
        CurrentBB = PredBB;
        CurrentInst = CurrentBB->getTerminator();
      }
    }
    if (InsertPoint != ReadCall) {
      ReadCall->moveBefore(InsertPoint->getIterator());
      return true;
    }
    return false;
  }

  bool sinkWrite(CallInst *WriteCall, AAResults &AA, const DataLayout &DL) {
    if (!WriteCall->use_empty()) return false;
    
    IOArgs Args = getIOArguments(WriteCall);
    if (!Args.Buffer) return false;
    
    MemoryLocation SrcLoc(Args.Buffer, LocationSize::beforeOrAfterPointer());
    Instruction *InsertPoint = WriteCall;
    Instruction *CurrentInst = WriteCall->getNextNode();
    
    while (CurrentInst) {
      if (CurrentInst->isTerminator() || isa<PHINode>(CurrentInst)) break;
      if (CurrentInst->mayThrow()) break;
      
      if (auto *CI = dyn_cast<CallInst>(CurrentInst)) {
        if (CI->getIntrinsicID() == Intrinsic::lifetime_end || 
            CI->getIntrinsicID() == Intrinsic::lifetime_start) {
	  break;
        }
        if (getIOArguments(CI).Type != IOArgs::NONE) break;
      }
      
      if (CurrentInst->mayReadOrWriteMemory()) {
        if (isModSet(AA.getModRefInfo(CurrentInst, SrcLoc))) break;
        MemoryLocation TargetLoc(Args.Target, LocationSize::beforeOrAfterPointer());
        if (isModSet(AA.getModRefInfo(CurrentInst, TargetLoc))) break;
      }
      
      InsertPoint = CurrentInst;
      CurrentInst = CurrentInst->getNextNode();
    }
    
    if (InsertPoint != WriteCall) {
      WriteCall->moveAfter(InsertPoint);
      return true;
    }
    return false;
  }
  
}

struct IOOptimisationPass : public PassInfoMixin<IOOptimisationPass> {
  IOOptimisationPass() {
    // Dynamically tune the pass during LTO based on environment variables
    IOBatchThreshold = getEnvOrDefault("IO_BATCH_THRESHOLD", 4);
    IOShadowBufferSize = getEnvOrDefault("IO_SHADOW_BUFFER_MAX", 4096);
    IOHighWaterMark = getEnvOrDefault("IO_HIGH_WATER_MARK", 65536);
  }

  bool optimiseLoopIO(Loop *L, ScalarEvolution &SE, const DataLayout &DL) {
    BasicBlock *ExitBB = L->getExitBlock();
    BasicBlock *Preheader = L->getLoopPreheader();
    if (!ExitBB || !Preheader) return false;

    unsigned TripCount = SE.getSmallConstantTripCount(L);
    if (TripCount == 0) return false;

    bool LoopChanged = false;

    // Inside optimiseLoopIO loop
    for (BasicBlock *BB : L->blocks()) {
      // Allow the block to have a branch if one side is a "Side-Exit" (return)
      // and the other side stays within the loop.
      Instruction *Term = BB->getTerminator();
      if (auto *BI = dyn_cast<BranchInst>(Term)) {
	if (BI->isConditional()) {
	  bool LeadsToExit = false;
	  for (BasicBlock *Succ : BI->successors()) {
	    if (L->isLoopExiting(Succ) && isa<ReturnInst>(Succ->getTerminator())) {
	      LeadsToExit = true;
	    }
	  }
	  // If this is just a standard error check, it's safe to ignore for hoisting
	  if (!LeadsToExit) continue; 
	}
      }

      for (Instruction &I : llvm::make_early_inc_range(*BB)) {
	if (auto *Call = dyn_cast<CallInst>(&I)) {
	  IOArgs Args = getIOArguments(Call);
            
	  if (Args.Type == IOArgs::POSIX_WRITE || Args.Type == IOArgs::POSIX_READ) {
                
	    auto *ConstLen = dyn_cast<ConstantInt>(Args.Length);
	    if (!ConstLen) continue;

	    uint64_t ElementSize = ConstLen->getZExtValue();
	    uint64_t TotalBytes = ElementSize * TripCount;

	    if (TotalBytes > IOHighWaterMark) continue;

	    if (!L->isLoopInvariant(Args.Target)) continue;

	    const SCEV *PtrSCEV = SE.getSCEV(Args.Buffer);
	    Value *BasePointer = nullptr;

	    if (auto *AddRec = dyn_cast<SCEVAddRecExpr>(PtrSCEV)) {
	      if (AddRec->getLoop() == L) {
		if (auto *StepConst = dyn_cast<SCEVConstant>(AddRec->getStepRecurrence(SE))) {
		  if (StepConst->getValue()->getZExtValue() == ElementSize) {
		    SCEVExpander Expander(SE, DL, "io.base.expander");
		    BasePointer = Expander.expandCodeFor(AddRec->getStart(), Args.Buffer->getType(), Preheader->getTerminator());
		  }
		}
	      }
	    } else if (SE.isLoopInvariant(PtrSCEV, L)) {
	      SCEVExpander Expander(SE, DL, "io.base.expander");
	      BasePointer = Expander.expandCodeFor(PtrSCEV, Args.Buffer->getType(), Preheader->getTerminator());
	    }

	    if (!BasePointer) continue;

	    IRBuilder<> ExitBuilder(&*ExitBB->getFirstInsertionPt());
	    Value *TotalLenVal = ExitBuilder.getIntN(Args.Length->getType()->getIntegerBitWidth(), TotalBytes);
              
	    std::vector<Value *> NewArgs = {Args.Target, BasePointer, TotalLenVal};
	    ExitBuilder.CreateCall(Call->getCalledFunction(), NewArgs);
 
	    if (!Call->use_empty()) {
	      Call->replaceAllUsesWith(Args.Length);
	    }


	    errs() << "[IOOpt] SUCCESS: Hoisted Loop POSIX I/O! Scaled " << ElementSize 
		   << " bytes * " << TripCount << " iterations = " << TotalBytes << " bytes at loop exit.\n";

	    Call->eraseFromParent();
	    LoopChanged = true;
	  }
	}
      }
    }
    return LoopChanged;
  }

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    errs() << "[IOOpt] Analysing function: " << F.getName() << "\n";
    bool Changed = false;

    AAResults &AA = FAM.getResult<AAManager>(F);
    const DataLayout &DL = F.getParent()->getDataLayout();
    LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
    ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);

    for (Loop *L : LI.getLoopsInPreorder()) {
      if (optimiseLoopIO(L, SE, DL)) Changed = true;
    }

    for (BasicBlock &BB : F) {
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
	if (auto *Call = dyn_cast<CallInst>(&I)) {
	  IOArgs CArgs = getIOArguments(Call);
	  if (CArgs.Type == IOArgs::POSIX_READ || CArgs.Type == IOArgs::C_FREAD || CArgs.Type == IOArgs::POSIX_PREAD) {
	    if (hoistRead(Call, AA, DL)) Changed = true;
	  }
	}
      }
    }

    for (BasicBlock &BB : F) {
      for (Instruction &I : llvm::make_early_inc_range(llvm::reverse(BB))) {
	if (auto *Call = dyn_cast<CallInst>(&I)) {
	  IOArgs CArgs = getIOArguments(Call);
	  if (CArgs.Type == IOArgs::POSIX_WRITE || CArgs.Type == IOArgs::C_FWRITE || 
	      CArgs.Type == IOArgs::CXX_WRITE || CArgs.Type == IOArgs::POSIX_PWRITE) {
	    if (sinkWrite(Call, AA, DL)) Changed = true;
	  }
	}
      }
    }

    // Phase 4: Cross-Block Stitching Batcher
    std::set<BasicBlock*> Visited;
    for (BasicBlock &BB : F) {
      if (Visited.count(&BB)) continue;

      std::vector<CallInst*> IOBatch;
      BasicBlock *CurrentBB = &BB;

      while (CurrentBB) {
	Visited.insert(CurrentBB);
	bool HazardInBlock = false;

	for (Instruction &I : llvm::make_early_inc_range(*CurrentBB)) {
	  if (auto *Call = dyn_cast<CallInst>(&I)) {
	    if (getIOArguments(Call).Type != IOArgs::NONE) {
	      if (isSafeToAddToBatch(IOBatch, Call, AA, DL, SE)) {
		IOBatch.push_back(Call);
	      } else {
		HazardInBlock = true;
		break; 
	      }
	    }
	  }
	}

	if (HazardInBlock) break;

	// Stitching Logic: Follow the path that continues I/O
	Instruction *Term = CurrentBB->getTerminator();
	BasicBlock *NextBB = nullptr;
	if (auto *BI = dyn_cast<BranchInst>(Term)) {
	  if (BI->isConditional()) {
	    // Enable the following  pattern: if (err) return; else continue;
	    // We follow the successor that does not return.
	    for (BasicBlock *Succ : BI->successors()) {
	      if (!isa<ReturnInst>(Succ->getTerminator())) {
		NextBB = Succ;
		break;
	      }
	    }
	  } else {
	    NextBB = BI->getSuccessor(0);
	  }
	}

	// Only follow if the next block has only one predecessor (us)
	if (NextBB && NextBB->getSinglePredecessor() == CurrentBB && !Visited.count(NextBB))
	  CurrentBB = NextBB;
	else
	  CurrentBB = nullptr;
      }

      if (flushBatch(IOBatch, F.getParent(), SE)) Changed = true;
    }
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};


struct IOWrapperInlinePass : public PassInfoMixin<IOWrapperInlinePass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    bool Changed = false;
        
    for (Function &F : M) {
      if (F.isDeclaration()) continue;

      unsigned InstCount = 0;
      bool HasIO = false;

      for (BasicBlock &BB : F) {
	for (Instruction &I : BB) {
	  InstCount++;
	  if (auto *Call = dyn_cast<CallInst>(&I)) {
	    if (Function *Callee = Call->getCalledFunction()) {
	      StringRef Name = Callee->getName();
	      if (Name == "write" || Name == "writev" || Name == "write64" ||
		  Name == "read" || Name == "readv" || Name == "read64" ||
		  Name == "fwrite" || Name == "fread" ||
		  Name == "pwrite" || Name == "pread" || 
		  Name == "pwrite64" || Name == "pread64" || 
		  Name == "MPI_File_write_at" || Name == "PMPI_File_write_at" ||
		  Name == "MPI_File_read_at" || Name == "PMPI_File_read_at") {
		HasIO = true;
	      }
	    }
	  }
	}
      }

      if (HasIO && InstCount < 10 && !F.hasFnAttribute(Attribute::NoInline)) {
	errs() << "[IOOpt-Injector] Tagging '" << F.getName() << "' for aggressive LTO inlining...\n";
	F.addFnAttr(Attribute::AlwaysInline);
	Changed = true;
      }
    }
        
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "IOOpt", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      
      PB.registerPipelineParsingCallback(
					 [](StringRef Name, FunctionPassManager &FPM,
					    ArrayRef<PassBuilder::PipelineElement>) {
					   if (Name == "io-opt") {
					     FPM.addPass(IOOptimisationPass()); 
					     return true;
					   }
					   return false;
					 });
      
      PB.registerPipelineStartEPCallback(
					 [](ModulePassManager &MPM, OptimizationLevel Level) {
					   MPM.addPass(IOWrapperInlinePass());
					 });
      
      PB.registerOptimizerLastEPCallback(
					 [](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase Phase) {
					   MPM.addPass(createModuleToFunctionPassAdaptor(IOOptimisationPass())); 
					 });
      
      PB.registerFullLinkTimeOptimizationLastEPCallback(
							[](ModulePassManager &MPM, OptimizationLevel Level) {
							  MPM.addPass(createModuleToFunctionPassAdaptor(IOOptimisationPass())); 
							});
    }};
}
