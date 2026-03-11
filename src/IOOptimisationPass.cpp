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
#include "llvm/IR/Dominators.h"

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


  bool checkAdjacency(Value *Buf1, Value *Len1, Value *Buf2, const DataLayout &DL, ScalarEvolution *SE, bool AllowGaps = false) {
    if (SE) {
      const SCEV *Ptr1 = SE->getSCEV(Buf1);
      const SCEV *Ptr2 = SE->getSCEV(Buf2);
      const SCEV *Size1 = SE->getSCEV(Len1);

      if (!isa<SCEVCouldNotCompute>(Ptr1) && !isa<SCEVCouldNotCompute>(Ptr2) && !isa<SCEVCouldNotCompute>(Size1)) {
	const SCEV *ExtendedSize = SE->getTruncateOrZeroExtend(Size1, Ptr1->getType());
	const SCEV *ExpectedNext = SE->getAddExpr(Ptr1, ExtendedSize);

	if (ExpectedNext == Ptr2) return true; 

	// Only allow gaps if the caller explicitly asks (ShadowBuffer path)
	if (AllowGaps) {
	  const SCEV *Distance = SE->getMinusSCEV(Ptr2, ExpectedNext);
	  if (auto *ConstDist = dyn_cast<SCEVConstant>(Distance)) {
	    int64_t Gap = ConstDist->getValue()->getSExtValue();
	    if (Gap >= 0 && Gap < 1024) return true;
	  }
	}
      }
    }

    // Fallback logic
    APInt Off1(DL.getIndexTypeSizeInBits(Buf1->getType()), 0);
    const Value *Base1 = Buf1->stripAndAccumulateConstantOffsets(DL, Off1, true);
    APInt Off2(DL.getIndexTypeSizeInBits(Buf2->getType()), 0);
    const Value *Base2 = Buf2->stripAndAccumulateConstantOffsets(DL, Off2, true);

    if (Base1 && Base1 == Base2) {
      if (auto *CLen = dyn_cast<ConstantInt>(Len1)) {
	uint64_t End1 = Off1.getZExtValue() + CLen->getZExtValue();
	uint64_t Start2 = Off2.getZExtValue();
	if (End1 == Start2) return true;
	if (AllowGaps && Start2 > End1 && (Start2 - End1) < 1024) return true;
      }
    }

    return false;
  }

  bool isSafeToAddToBatch(const std::vector<CallInst*> &Batch, CallInst *NewCall, AAResults &AA, const DataLayout &DL, ScalarEvolution &SE, DominatorTree &DT) {
    errs() << "[IOOpt-Debug] Attempting to add " << *NewCall << " to batch of size " << Batch.size() << "\n";
    if (Batch.empty()) return true;

    CallInst *LastCall = Batch.back();

    if (!DT.dominates(LastCall, NewCall)) {
        errs() << "[IOOpt-Debug] Batch Break: CFG Dominance violation.\n";
        return false;
    }

    if (LastCall->getCalledFunction() != NewCall->getCalledFunction()) {
      errs() << "[IOOpt-Debug] Batch Break: Function mismatch.\n";
      return false;
    }

    IOArgs FirstArgs = getIOArguments(Batch.front());
    IOArgs LastArgs = getIOArguments(LastCall);
    IOArgs NewArgs = getIOArguments(NewCall);

    if (LastCall->getCalledFunction() != NewCall->getCalledFunction()) {
      errs() << "[IOOpt-Debug] Batch Break: Function mismatch.\n";
      return false;
    }
 
    // C++ ostream::write returns 'this' (the stream pointer).
    // If NewCall's first argument is the result of LastCall, it's a "chain," not a hazard.
    bool IsCXXChain = false;
    if (NewArgs.Type == IOArgs::CXX_WRITE) {
      if (NewCall->getArgOperand(0) == LastCall) {
	IsCXXChain = true;
      }
    }

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
	  // Must be contiguous on disk for pwrite/pread
	  const SCEV *ExpectedNext = SE.getAddExpr(SLast, SE.getTruncateOrZeroExtend(SLen, SLast->getType()));
	  if (ExpectedNext == SNew) {
	    isContiguous = true;
	  }
        }
      }
    
      // Fallback: raw Constants
      if (!isContiguous) {
        if (auto *CLastOff = dyn_cast<ConstantInt>(LastOffset)) {
	  if (auto *CNewOff = dyn_cast<ConstantInt>(NewOffset)) {
	    if (auto *CLen = dyn_cast<ConstantInt>(LastLen)) {
	      if (CLastOff->getZExtValue() + CLen->getZExtValue() == CNewOff->getZExtValue()) {
		isContiguous = true;
	      }
	    }
	  }
        }
      }
    
      // Do not allow gaps for explicit offsets
      if (!isContiguous) return false;

    } else if (NewArgs.Type == IOArgs::MPI_READ_AT || NewArgs.Type == IOArgs::MPI_WRITE_AT) {

      // If the application provides different status pointers for each call,
      // we cannot safely batch them without causing uninitialized memory reads
      if (LastCall->getArgOperand(5) != NewCall->getArgOperand(5)) {
          return false;
      }

      // MPI Smart Datatype Checking (resolve opaque pointer loads)
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

      if (SE.isSCEVable(LastOffset->getType()) && SE.isSCEVable(NewOffset->getType())) {
        const SCEV *SLast = SE.getSCEV(LastOffset);
        const SCEV *SNew = SE.getSCEV(NewOffset);
        const SCEV *SCount = SE.getTruncateOrZeroExtend(SE.getSCEV(LastCount), SLast->getType());

        if (!isa<SCEVCouldNotCompute>(SLast) && !isa<SCEVCouldNotCompute>(SNew)) {
	  const SCEV *ExpectedNext = SE.getAddExpr(SLast, SCount);
	  if (ExpectedNext == SNew) {
	    isContiguous = true;
	  } else {
	    // Gap Tolerance: Allow small jumps (e.g., < 1024 bytes) for Shadow Buffering
	    const SCEV *GapSCEV = SE.getMinusSCEV(SNew, ExpectedNext);
	    if (auto *ConstGap = dyn_cast<SCEVConstant>(GapSCEV)) {
	      int64_t GapVal = ConstGap->getValue()->getSExtValue();
	      if (GapVal > 0 && GapVal < 1024) isContiguous = true;
	    }
	  }
        }
      }

      // Fallback for raw constants
      if (!isContiguous) {
        if (auto *CLastOff = dyn_cast<ConstantInt>(LastOffset)) {
	  if (auto *CNewOff = dyn_cast<ConstantInt>(NewOffset)) {
	    if (auto *CLen = dyn_cast<ConstantInt>(LastCount)) {
	      uint64_t Gap = CNewOff->getZExtValue() - (CLastOff->getZExtValue() + CLen->getZExtValue());
	      if (Gap < 1024) isContiguous = true;
	    }
	  }
        }
      }

      if (!isContiguous) return false;

    } else if (NewArgs.Type == IOArgs::POSIX_READ || NewArgs.Type == IOArgs::POSIX_WRITE) {
      // Implicit offsets natively guarantee disk contiguity if File Descriptors match.
      // We check the Target (the FD) for equality.
      bool FDsMatch = (NewArgs.Target == LastArgs.Target);
    
      // Smart FD resolution: check if they load from the same memory location
      if (!FDsMatch) {
        auto *L1 = dyn_cast<LoadInst>(LastArgs.Target);
        auto *L2 = dyn_cast<LoadInst>(NewArgs.Target);
        if (L1 && L2 && L1->getPointerOperand() == L2->getPointerOperand()) {
	  FDsMatch = true;
        }
      }

      if (!FDsMatch) return false; 
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

    // Hazard Scanning Loop (Cross-Block Support)
    Instruction *CurrInst = LastCall->getNextNode();
    BasicBlock *CurrBB = LastCall->getParent();

    while (CurrInst != NewCall) {
      if (!CurrInst) {
	// If we hit the end of the block, move to the next sequential block
	CurrBB = CurrBB->getNextNode();
	if (!CurrBB) return false; // Reached end of function safely
	CurrInst = &CurrBB->front();
	continue;
      }

      // Opaque Barriers and Lifetimes
      if (auto *CI = dyn_cast<CallInst>(CurrInst)) {
	if (CI->getIntrinsicID() == Intrinsic::lifetime_end || 
	    CI->getIntrinsicID() == Intrinsic::lifetime_start) {
	  return false;
	}
	if (getIOArguments(CI).Type != IOArgs::NONE) return false;
	if (!CI->onlyReadsMemory() && !CI->doesNotAccessMemory()) return false;
      }

      if (CurrInst->mayReadOrWriteMemory()) {
	if (isModSet(AA.getModRefInfo(CurrInst, NewLoc))) return false;

	for (CallInst *BatchedCall : Batch) {
	  IOArgs BArgs = getIOArguments(BatchedCall);
	  MemoryLocation BLoc = getPreciseLoc(BArgs.Buffer, BArgs.Length);
	  if (isModSet(AA.getModRefInfo(CurrInst, BLoc))) return false;
	}

	if (FirstArgs.Target->getType()->isPointerTy()) {
	  MemoryLocation TargetLoc(FirstArgs.Target, LocationSize::beforeOrAfterPointer());
	  if (isModSet(AA.getModRefInfo(CurrInst, TargetLoc))) return false;
	}
          
	if (Load1) {
	  MemoryLocation FdLoc(Load1->getPointerOperand(), LocationSize::beforeOrAfterPointer());
	  if (isModSet(AA.getModRefInfo(CurrInst, FdLoc))) return false;
	}
      }
      CurrInst = CurrInst->getNextNode();
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

  // --- Helper 1: Detects fixed-distance strides (Struct arrays/MPI Columnar) ---
  bool isStridedPattern(const std::vector<CallInst*> &Batch, const DataLayout &DL, ScalarEvolution *SE) {
    if (!SE || Batch.size() < 2) return false;

    // We need to compare the distance between the buffers of each call
    const SCEV *Ptr0 = SE->getSCEV(getIOArguments(Batch[0]).Buffer);
    const SCEV *Ptr1 = SE->getSCEV(getIOArguments(Batch[1]).Buffer);
    const SCEV *Stride = SE->getMinusSCEV(Ptr1, Ptr0);

    // If the stride is 0 or couldn't be computed, it's not a valid stride
    if (isa<SCEVCouldNotCompute>(Stride) || Stride->isZero()) return false;

    for (size_t i = 1; i < Batch.size() - 1; ++i) {
      const SCEV *CurrentPtr = SE->getSCEV(getIOArguments(Batch[i]).Buffer);
      const SCEV *NextPtr = SE->getSCEV(getIOArguments(Batch[i+1]).Buffer);
      if (SE->getMinusSCEV(NextPtr, CurrentPtr) != Stride) return false;
    }
    return true;
  }

  // --- Helper 2: Calculates the absolute byte-span for Shadow Buffering ---
  bool calculateTotalRange(const std::vector<CallInst*> &Batch, ScalarEvolution *SE, 
			   uint64_t &MinOff, uint64_t &MaxOff) {
    if (!SE || Batch.empty()) return false;

    bool First = true;
    for (CallInst *CI : Batch) {
      IOArgs Args = getIOArguments(CI);
      // For pwrite/pread, the offset is operand 3. For MPI_Write_at, it's operand 1.
      Value *OffVal = (Args.Type == IOArgs::POSIX_PREAD || Args.Type == IOArgs::POSIX_PWRITE) 
	? CI->getArgOperand(3) : CI->getArgOperand(1);
        
      const SCEV *S_Off = SE->getSCEV(OffVal);
      const SCEV *S_Len = SE->getTruncateOrZeroExtend(SE->getSCEV(Args.Length), S_Off->getType());
        
      auto *C_Off = dyn_cast<SCEVConstant>(S_Off);
      auto *C_Len = dyn_cast<SCEVConstant>(S_Len);

      if (!C_Off || !C_Len) return false; 

      uint64_t Start = C_Off->getValue()->getZExtValue();
      uint64_t End = Start + C_Len->getValue()->getZExtValue();

      if (First) {
	MinOff = Start; MaxOff = End;
	First = false;
      } else {
	MinOff = std::min(MinOff, Start);
	MaxOff = std::max(MaxOff, End);
      }
    }
    return true;
  }

  // ====================================================================
  // Classifier: Analyes the batch and decides the best strategy
  // ====================================================================
  IOPattern classifyBatch(const std::vector<CallInst*> &Batch, const DataLayout &DL, 
			  uint64_t &OutTotalRange, ScalarEvolution *SE) {
    if (Batch.size() < 2) return IOPattern::Unprofitable;

    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isRead = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD || FirstArgs.Type == IOArgs::POSIX_PREAD);

    // Physically contiguous
    bool StrictPhysical = true;
    for (size_t i = 0; i < Batch.size() - 1; ++i) {
      if (!checkAdjacency(getIOArguments(Batch[i]).Buffer, getIOArguments(Batch[i]).Length, 
			  getIOArguments(Batch[i+1]).Buffer, DL, SE, false)) {
	StrictPhysical = false;
	break;
      }
    }
    if (StrictPhysical) return IOPattern::Contiguous;

    // Strided Pattern
    if (isStridedPattern(Batch, DL, SE)) {
      // Grab the length of the first call's buffer
      if (auto *ConstLen = dyn_cast<ConstantInt>(getIOArguments(Batch.front()).Length)) {
	OutTotalRange = ConstLen->getZExtValue();
	return IOPattern::Strided;
      }
    }

    // Vectored I/O
    size_t DynamicThreshold = IOBatchThreshold; // Defaults to 4
    
    if (isRead) {
      DynamicThreshold = 2; 
    } else {
      Function *F = Batch.back()->getFunction();
      if (F->getInstructionCount() > 150) {
	DynamicThreshold = 3;
      } else {
	// Check if we have constant sizes that add up to > 128 bytes
	uint64_t TotalBytes = 0;
	bool AllConstant = true;
	for (CallInst *C : Batch) {
	  if (auto *CI = dyn_cast<ConstantInt>(getIOArguments(C).Length)) {
	    TotalBytes += CI->getZExtValue();
	  } else {
	    AllConstant = false;
	    break;
	  }
	}
	if (AllConstant && TotalBytes > 128) {
	  DynamicThreshold = 3;
	}
      }
    }

    if (Batch.size() >= DynamicThreshold) {
      // Only raw POSIX supports Vectored I/O (readv/writev/preadv/pwritev)
      // fwrite, CXX_WRITE, and MPI must fall through to ShadowBuffering
      if (FirstArgs.Type == IOArgs::POSIX_READ || 
          FirstArgs.Type == IOArgs::POSIX_WRITE || 
          FirstArgs.Type == IOArgs::POSIX_PREAD || 
          FirstArgs.Type == IOArgs::POSIX_PWRITE) {
        return IOPattern::Vectored;
      }
    }

    // Shadow buffer (fallback for small writes that missed the Vectored threshold)
    if (FirstArgs.Type == IOArgs::POSIX_WRITE || 
        FirstArgs.Type == IOArgs::C_FWRITE || 
        FirstArgs.Type == IOArgs::CXX_WRITE ||
        FirstArgs.Type == IOArgs::MPI_WRITE_AT) { 
        
      uint64_t TotalConstSize = 0;
      bool AllSizesConstant = true;
        
      for (CallInst *C : Batch) {
	if (auto *CI = dyn_cast<ConstantInt>(getIOArguments(C).Length)) {
	  TotalConstSize += CI->getZExtValue();
	} else {
	  AllSizesConstant = false;
	  break;
	}
      }
        
      // If all sizes are known and fit comfortably on the stack
      if (AllSizesConstant && TotalConstSize > 0 && TotalConstSize <= IOShadowBufferSize) {
	OutTotalRange = TotalConstSize;
	return IOPattern::ShadowBuffer;
      }
    }

    return IOPattern::Unprofitable;
  }

  // ====================================================================
  // The Router: Executes the IR transformations based on the Classifier
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

    // This will hold our new optimized I/O instruction
    CallInst *MergedCall = nullptr;

    switch (Pattern) {
    case IOPattern::Contiguous: {
      Value *TotalLen = FirstArgs.Length;
      for (size_t i = 1; i < Batch.size(); ++i) {
	TotalLen = Builder.CreateAdd(TotalLen, getIOArguments(Batch[i]).Length, "sum.len");
      }
              
      std::vector<Value *> NewArgs;
      if (FirstArgs.Type == IOArgs::MPI_WRITE_AT || FirstArgs.Type == IOArgs::MPI_READ_AT) {
	NewArgs = { Batch[0]->getArgOperand(0), Batch[0]->getArgOperand(1), FirstArgs.Buffer, TotalLen, Batch[0]->getArgOperand(4), Batch[0]->getArgOperand(5) };
      } else if (FirstArgs.Type == IOArgs::C_FWRITE || FirstArgs.Type == IOArgs::C_FREAD) {
	NewArgs = {FirstArgs.Buffer, Batch[0]->getArgOperand(1), TotalLen, FirstArgs.Target};
      } else if (isExplicit) {
	NewArgs = {FirstArgs.Target, FirstArgs.Buffer, TotalLen, Batch[0]->getArgOperand(3)};
      } else {
	NewArgs = {FirstArgs.Target, FirstArgs.Buffer, TotalLen};
      }
      MergedCall = Builder.CreateCall(Batch[0]->getCalledFunction(), NewArgs);
      errs() << "[IOOpt] SUCCESS: N-Way merged " << Batch.size() << " contiguous " << (isRead ? "reads" : "writes") << "!\n";
      break;
    }

    case IOPattern::Strided: {
      // Validate Size (Prevent i0)
      unsigned ElementBytes = TotalConstSize; 
      if (ElementBytes == 0) {
        errs() << "[IOOpt-Error] Strided element size is 0! Aborting.\n";
        return false;
      }
    
      unsigned NumElements = Batch.size();
      Type *ElementTy = Builder.getIntNTy(ElementBytes * 8); 
      auto *VecTy = FixedVectorType::get(ElementTy, NumElements);
    
      // Build the Gather Vector
      Value *GatherVec = PoisonValue::get(VecTy);
      for (unsigned i = 0; i < NumElements; ++i) {
        IOArgs Args = getIOArguments(Batch[i]);
        // Ensure we load from the buffer with the correct type
        LoadInst *LoadedVal = Builder.CreateLoad(ElementTy, Args.Buffer, "strided.load");
        GatherVec = Builder.CreateInsertElement(GatherVec, LoadedVal, Builder.getInt32(i), "gather.insert");
      }
    
      // Clean Alloca (Prevent swifterror/huge alignment)
      Function *F = Batch.back()->getFunction();
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());
    
      // Use the simplest Alloca signature: Type, AddrSpace, Name
      AllocaInst *ContiguousBuf = EntryBuilder.CreateAlloca(VecTy, nullptr, "simd.shadow.buf");
      // Set a sane alignment (e.g., 16 bytes for SIMD)
      ContiguousBuf->setAlignment(Align(16));
    
      Builder.CreateStore(GatherVec, ContiguousBuf);
    
      // Issue the single Write
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
      errs() << "[IOOpt] SUCCESS: SIMD Gathered " << NumElements << " strided writes into 1!\n";
      break;
    }
    case IOPattern::ShadowBuffer: {
      Function *F = Batch.back()->getFunction();
      // Move the Alloca to the Entry Block (standard LLVM practice)
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());

      Type *Int8Ty = Builder.getInt8Ty();
      ArrayType *ShadowArrTy = ArrayType::get(Int8Ty, TotalConstSize);
      AllocaInst *ShadowBuf = EntryBuilder.CreateAlloca(ShadowArrTy, nullptr, "shadow.buf");
      ShadowBuf->setAlignment(Align(16)); // Sane alignment for general use

      // Pack the data into the buffer
      uint64_t CurrentOffset = 0;
      for (size_t i = 0; i < Batch.size(); ++i) {
        IOArgs Args = getIOArguments(Batch[i]);

        // Calculate the destination pointer inside our shadow buffer
        Value *DestPtr = Builder.CreateInBoundsGEP(
						   ShadowArrTy, ShadowBuf,
						   {Builder.getInt32(0), Builder.getInt32(CurrentOffset)},
						   "shadow.ptr"
						   );

        // We use the length from the specific call
        Value *Len = Args.Length;
        Builder.CreateMemCpy(DestPtr, Align(1), Args.Buffer, Align(1), Len);

        // Update the tracking offset for the next piece of data
        if (auto *C = dyn_cast<ConstantInt>(Len)) {
	  CurrentOffset += C->getZExtValue();
        }
      }

      // Emit the single merged write
      Value *TotalLenVal = Builder.getIntN(
					   FirstArgs.Length->getType()->getIntegerBitWidth(),
					   TotalConstSize
					   );

      // Cast shadow.buf to ptr for the write call
      Value *BufPtr = Builder.CreatePointerCast(ShadowBuf, Builder.getPtrTy());

      std::vector<Value *> NewArgs;

      if (FirstArgs.Type == IOArgs::MPI_WRITE_AT) {
        NewArgs = {
	  Batch[0]->getArgOperand(0), // fh
	  Batch[0]->getArgOperand(1), // offset
	  BufPtr,                     // coalesced stack array
	  TotalLenVal,                // summed count
	  Batch[0]->getArgOperand(4), // datatype
	  Batch[0]->getArgOperand(5)  // status
        };
      } else if (FirstArgs.Type == IOArgs::C_FWRITE) {
        // fwrite(buf, size, count, FILE*)
        NewArgs = {BufPtr, Batch[0]->getArgOperand(1), TotalLenVal, FirstArgs.Target};
      } else if (FirstArgs.Type == IOArgs::POSIX_PWRITE) {
        // pwrite(fd, buf, count, offset)
        NewArgs = {FirstArgs.Target, BufPtr, TotalLenVal, Batch[0]->getArgOperand(3)};
      } else {
        // Implicit writes: write(fd, buf, count)
        NewArgs = {FirstArgs.Target, BufPtr, TotalLenVal};
      }

      Builder.CreateCall(Batch[0]->getCalledFunction(), NewArgs);
      errs() << "[IOOpt] SUCCESS: Shadow Buffered " << Batch.size() << " writes into 1 (" << TotalConstSize << " bytes)!\n";
      break;
    }

    case IOPattern::Vectored: {
      Type *Int32Ty = Builder.getInt32Ty();
      Type *PtrTy = PointerType::getUnqual(M->getContext());
      Type *SizeTy = DL.getIntPtrType(M->getContext());
            
      StringRef FuncName = isRead ? (isExplicit ? "preadv" : "readv") : (isExplicit ? "pwritev" : "writev");
      FunctionType *VecTy = isExplicit ? 
	FunctionType::get(SizeTy, {Int32Ty, PtrTy, Int32Ty, Batch[0]->getArgOperand(3)->getType()}, false) :
	FunctionType::get(SizeTy, {Int32Ty, PtrTy, Int32Ty}, false);
            
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
           
      // Decay the array pointer [N x iovec]* to iovec*
      Value *IovBasePtr = Builder.CreateInBoundsGEP(
						    IovArrayTy, IovArray, 
						    {Builder.getInt32(0), Builder.getInt32(0)}, 
						    "iovec.base.ptr"
						    );
 
      Value *Fd = Builder.CreateIntCast(FirstArgs.Target, Int32Ty, false);
      if (isExplicit) {
	MergedCall = Builder.CreateCall(VecFunc, {Fd, IovArray, Builder.getInt32(Batch.size()), Batch[0]->getArgOperand(3)});
      } else {
	MergedCall = Builder.CreateCall(VecFunc, {Fd, IovArray, Builder.getInt32(Batch.size())});
      }
      errs() << "[IOOpt] SUCCESS: N-Way converted " << Batch.size() << " " << (isRead ? "reads" : "writes") << " to " << FuncName << "!\n";
      break;
    }
    default: break;
    }

    // Inside flushBatch cleanup loop
    // Inside flushBatch cleanup loop
    for (CallInst *C : Batch) {
      if (!C->use_empty()) {
        IOArgs CArgs = getIOArguments(C);
        Value *Rep;
        
        if (CArgs.Type == IOArgs::MPI_WRITE_AT || CArgs.Type == IOArgs::MPI_READ_AT) {
            // MPI expects an error code (0 = MPI_SUCCESS)
            Rep = Builder.getInt32(0);
        } else if (CArgs.Type == IOArgs::C_FWRITE || CArgs.Type == IOArgs::C_FREAD) {
            // C Standard Library expects the 'count' argument (Operand 2)
            Rep = C->getArgOperand(2); 
            if (C->getType() != Rep->getType()) {
                Rep = Builder.CreateIntCast(Rep, C->getType(), false);
            }
        } else {
            // POSIX expects the total byte length
            Rep = CArgs.Length;
            if (C->getType() != Rep->getType()) {
                Rep = Builder.CreateIntCast(Rep, C->getType(), false);
            }
        }
        
        C->replaceAllUsesWith(Rep);
      }
      C->dropAllReferences();
      C->eraseFromParent();
    }
    errs() << "[IOOpt] SUCCESS: Flushed batch of " << Batch.size() << " calls.\n"; 
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

      for (BasicBlock *BB : L->blocks()) {
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
      DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);

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
     
      // Phase 4: Function-level batching (Cross-Block)
      std::vector<CallInst*> IOBatch;
      uint64_t CurrentBatchBytes = 0;

      for (BasicBlock &BB : F) {
        for (Instruction &I : llvm::make_early_inc_range(BB)) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
              
            if (Function *CalleeF = Call->getCalledFunction()) {
	      StringRef FuncName = CalleeF->getName();
	      if (FuncName == "fsync" || FuncName == "fdatasync" || 
		  FuncName == "sync_file_range" || FuncName == "posix_fadvise" || 
		  FuncName == "posix_fadvise64" || FuncName == "madvise") {
                    
		// Flush immediately to prevent crossing the sync barrier
		if (flushBatch(IOBatch, F.getParent(), SE)) Changed = true;
		CurrentBatchBytes = 0;
		continue; 
	      }
            }

            IOArgs CArgs = getIOArguments(Call);

            bool isWrite = (CArgs.Type == IOArgs::POSIX_WRITE || CArgs.Type == IOArgs::C_FWRITE || 
                            CArgs.Type == IOArgs::CXX_WRITE || CArgs.Type == IOArgs::POSIX_PWRITE || 
                            CArgs.Type == IOArgs::MPI_WRITE_AT);
            
            bool isRead = (CArgs.Type == IOArgs::POSIX_READ || CArgs.Type == IOArgs::C_FREAD || 
                           CArgs.Type == IOArgs::POSIX_PREAD || CArgs.Type == IOArgs::MPI_READ_AT);

            if (isWrite || isRead) {
              uint64_t CallBytes = 4096; // Fallback for variable lengths
              if (auto *ConstLen = dyn_cast<ConstantInt>(CArgs.Length)) {
		CallBytes = ConstLen->getZExtValue();
              }

              // If we are switching between Read and Write, flush the current batch
              if (!IOBatch.empty()) {
                IOArgs BatchArgs = getIOArguments(IOBatch.front());
                bool BatchIsRead = (BatchArgs.Type == IOArgs::POSIX_READ || BatchArgs.Type == IOArgs::C_FREAD || 
                                    BatchArgs.Type == IOArgs::POSIX_PREAD || BatchArgs.Type == IOArgs::MPI_READ_AT);
                if (BatchIsRead != isRead) {
		  if (flushBatch(IOBatch, F.getParent(), SE)) Changed = true;
		  CurrentBatchBytes = 0;
                }
              }

              // Check for memory/disk hazards using our cross-block scanner
              if (isSafeToAddToBatch(IOBatch, Call, AA, DL, SE, DT)) {
                IOBatch.push_back(Call);
                CurrentBatchBytes += CallBytes;

                // High Water Mark constraint
                if (CurrentBatchBytes >= IOHighWaterMark) {
                  if (flushBatch(IOBatch, F.getParent(), SE)) Changed = true;
                  CurrentBatchBytes = 0;
                }
              } else {
                // Hazard detected. Flush the existing batch and start a new one.
                if (flushBatch(IOBatch, F.getParent(), SE)) Changed = true;
                IOBatch.push_back(Call);
                CurrentBatchBytes = CallBytes; 
              }
            }
          }
        }
      }
      
      if (flushBatch(IOBatch, F.getParent(), SE)) Changed = true;

      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    } 
     
  };
}

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
