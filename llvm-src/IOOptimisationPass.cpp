#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Transforms/Utils/LoopSimplify.h" 
#include "llvm/Transforms/Utils/LCSSA.h"        
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h" 
#include "llvm/ADT/Statistic.h"
#include "llvm/Config/abi-breaking.h"

// Clang and LLD do not export these symbols dynamically like 'opt' does.
// By defining them here, we satisfy the dynamic loader so our plugin 
// can be loaded during the Link-Time Optimization phase.
namespace llvm {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  int EnableABIBreakingChecks = 1;
#else
  int DisableABIBreakingChecks = 1;
#endif
}

#include <cstdlib>
#include <string>
#include <unordered_map>

using namespace llvm;

#define DEBUG_TYPE "io-opt"
STATISTIC(NumLoopsHoisted, "Number of dynamic loop I/Os hoisted safely");
STATISTIC(NumBatchesMerged, "Number of standard I/O batches merged");
STATISTIC(NumZeroCopy, "Number of zero-copy (splice/sendfile) optimizations");
STATISTIC(NumIPAInlines, "Number of inter-procedural I/O chains collapsed");
STATISTIC(NumFunctionsAnalyzed, "Number of functions analyzed by IOOpt");

static unsigned getEnvOrDefault(const char* Name, unsigned Default) {
  if (const char* Val = std::getenv(Name)) return std::stoi(Val);
  return Default;
}

struct IOConfig {
  unsigned BatchThreshold;
  unsigned ShadowBufferSize;
  unsigned HighWaterMark;
  unsigned MaxIov;
  bool EnableLogging;

  IOConfig() {
    BatchThreshold = getEnvOrDefault("IO_BATCH_THRESHOLD", 4);
    if(BatchThreshold <= 0) BatchThreshold = 4;
    ShadowBufferSize = getEnvOrDefault("IO_SHADOW_BUFFER_MAX", 4096);
    if(ShadowBufferSize <= 0) ShadowBufferSize = 4096;
    HighWaterMark = getEnvOrDefault("IO_HIGH_WATER_MARK", 65536);
    if(HighWaterMark <= 0) HighWaterMark = 65536;
    EnableLogging = getEnvOrDefault("IO_ENABLE_LOGGING", 1) != 0;
    MaxIov = getEnvOrDefault("IO_MAX_IOV", 1024);
    if (MaxIov <= 0) MaxIov = 1024;

  }
};

static IOConfig Config;

static void logMessage(const Twine &Msg) {
  if (Config.EnableLogging) errs() << Msg << "\n";
}

namespace {

  struct IOArgs {
    Value *Target; 
    Value *Buffer; 
    Value *Length; 
    enum { 
      NONE, C_FWRITE, C_FREAD, POSIX_WRITE, POSIX_READ, POSIX_PWRITE, POSIX_PREAD, 
      CXX_WRITE, CXX_READ, MPI_WRITE_AT, MPI_READ_AT, 
      SPLICE, SENDFILE, POSIX_PWRITEV, POSIX_PREADV, IO_SUBMIT, AIO_WRITE 
    } Type;
  };

  Value *getBaseFD(Value *Target) {
    if (!Target) return nullptr;
    if (Target->getType()->isPointerTy()) {
      return const_cast<Value*>(getUnderlyingObject(Target));
    }
    return Target; 
  }

  IOArgs getIOArguments(CallInst *Call, Function *F = nullptr) {
    auto getCStreamBytes = [](CallInst *CI) -> Value* {
      Value *Size = CI->getArgOperand(1);
      Value *Count = CI->getArgOperand(2);
      if (auto *CSize = dyn_cast<ConstantInt>(Size)) {
        if (CSize->getZExtValue() == 1) return Count;
        if (auto *CCount = dyn_cast<ConstantInt>(Count)) return ConstantInt::get(Count->getType(), CSize->getZExtValue() * CCount->getZExtValue());
      }
      if (auto *CCount = dyn_cast<ConstantInt>(Count)) {
        if (CCount->getZExtValue() == 1) return Size;
      }
      return nullptr; 
    };

    if (!F) F = Call->getCalledFunction();
    if (!F || !F->hasName() || !F->isDeclaration()) return {nullptr, nullptr, nullptr, IOArgs::NONE};

    std::string Demangled = llvm::demangle(F->getName().str());
    
    if (Demangled == "pread" || Demangled == "pread64") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PREAD};
    if (Demangled == "pwrite" || Demangled == "pwrite64") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PWRITE};
    if (Demangled == "write" || Demangled == "write64") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_WRITE};
    if (Demangled == "read" || Demangled == "read64")   return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_READ};
    
    if (Demangled == "fwrite" || Demangled == "efwrite") {
      Value *Bytes = getCStreamBytes(Call);
      return Bytes ? IOArgs{Call->getArgOperand(3), Call->getArgOperand(0), Bytes, IOArgs::C_FWRITE} : IOArgs{nullptr, nullptr, nullptr, IOArgs::NONE};
    }
    if (Demangled == "fread" || Demangled == "efread") {
      Value *Bytes = getCStreamBytes(Call);
      return Bytes ? IOArgs{Call->getArgOperand(3), Call->getArgOperand(0), Bytes, IOArgs::C_FREAD} : IOArgs{nullptr, nullptr, nullptr, IOArgs::NONE};
    }

    if (Demangled == "preadv" || Demangled == "preadv2") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PREADV};
    if (Demangled == "pwritev" || Demangled == "pwritev2") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PWRITEV};
    if (Demangled == "splice") return {Call->getArgOperand(2), Call->getArgOperand(0), Call->getArgOperand(4), IOArgs::SPLICE}; 
    if (Demangled == "sendfile" || Demangled == "sendfile64") return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(3), IOArgs::SENDFILE}; 
    if (Demangled == "io_submit") return {Call->getArgOperand(0), Call->getArgOperand(2), Call->getArgOperand(1), IOArgs::IO_SUBMIT}; 
    if (Demangled == "aio_write" || Demangled == "aio_write64") return {Call->getArgOperand(0), nullptr, nullptr, IOArgs::AIO_WRITE}; 
    if (Demangled == "MPI_File_write_at" || Demangled == "PMPI_File_write_at") return {Call->getArgOperand(0), Call->getArgOperand(2), Call->getArgOperand(3), IOArgs::MPI_WRITE_AT};
    if (Demangled == "MPI_File_read_at" || Demangled == "PMPI_File_read_at")  return {Call->getArgOperand(0), Call->getArgOperand(2), Call->getArgOperand(3), IOArgs::MPI_READ_AT};
    if ((Demangled.find("std::basic_ostream") != std::string::npos || Demangled.find("std::ostream") != std::string::npos) && Demangled.find("::write") != std::string::npos) {
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::CXX_WRITE};
    }
    if ((Demangled.find("std::basic_istream") != std::string::npos || Demangled.find("std::istream") != std::string::npos) && Demangled.find("::read") != std::string::npos) {
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::CXX_READ};
    }
    
    return {nullptr, nullptr, nullptr, IOArgs::NONE};
  }

  struct InterProceduralIOBatchingPass : public PassInfoMixin<InterProceduralIOBatchingPass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
      bool Changed = false;
      bool LocalChanged;
      
      do {
	LocalChanged = false;
	std::unordered_map<Function*, int> IOWrappers; 
          
	for (Function &F : M) {
	  if (F.isDeclaration()) continue;
	  int IOMapArg = -1;
	  bool hasIO = false;
	  unsigned instCount = 0;
              
	  for (BasicBlock &BB : F) {
	    for (Instruction &I : BB) {
	      instCount++;
	      if (auto *Call = dyn_cast<CallInst>(&I)) {
		Function *Callee = Call->getCalledFunction();
		IOArgs Args = getIOArguments(Call, Callee);
		if (Args.Type != IOArgs::NONE) {
		  hasIO = true;
		  if (auto *Arg = dyn_cast<Argument>(Args.Target)) IOMapArg = Arg->getArgNo();
		}
	      }
	    }
	  }
	  if (hasIO && instCount < 80 && IOMapArg != -1) IOWrappers[&F] = IOMapArg;
	}
          
	CallInst *TargetToInline = nullptr;
          
	for (Function &F : M) {
	  if (F.isDeclaration() || TargetToInline) break;
              
	  for (BasicBlock &BB : F) {
	    Value *LastIOFD = nullptr;
                  
	    for (Instruction &I : BB) {
	      if (auto *Call = dyn_cast<CallInst>(&I)) {
		Function *Callee = Call->getCalledFunction();
		if (!Callee) {
		  if (!Call->onlyReadsMemory()) LastIOFD = nullptr;
		  continue;
		}
                          
		IOArgs Args = getIOArguments(Call, Callee);
		if (Args.Type != IOArgs::NONE) {
		  LastIOFD = Args.Target;
		  continue;
		}
                          
		if (IOWrappers.count(Callee)) {
		  int ArgIdx = IOWrappers[Callee];
		  Value *PassedFD = Call->getArgOperand(ArgIdx);
		  if (LastIOFD != nullptr && PassedFD == LastIOFD) {
		    TargetToInline = Call;
		    break;
		  }
		  LastIOFD = PassedFD;
		} else {
		  if (!Call->onlyReadsMemory()) LastIOFD = nullptr;
		}
	      } else if (I.mayWriteToMemory()) {
		LastIOFD = nullptr; 
	      }
	    }
	    if (TargetToInline) break;
	  }
	}
          
	if (TargetToInline) {
	  // Grab the caller and callee before we destroy the call instruction via inlining
	  Function *Caller = TargetToInline->getFunction();
	  Function *Callee = TargetToInline->getCalledFunction();
              
	  std::string CallerName = Caller ? llvm::demangle(Caller->getName().str()) : "unknown";
	  std::string CalleeName = Callee ? llvm::demangle(Callee->getName().str()) : "unknown";

	  InlineFunctionInfo IFI;
	  if (InlineFunction(*TargetToInline, IFI).isSuccess()) {
	    LocalChanged = true;
	    Changed = true;
	    NumIPAInlines++;
                  
	    // This will specifically document when an I/O function from one file 
	    // is merged into a calling function from another file during LTO
	    logMessage("[IOOpt-LTO] SUCCESS: Inlined I/O wrapper '" + 
		       Twine(CalleeName) + "' into '" + Twine(CallerName) + "'.");
	  }
	}

          
      } while (LocalChanged);
      
      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };

  bool checkAdjacency(Value *Buf1, Value *Len1, Value *Buf2, const DataLayout &DL, ScalarEvolution *SE, bool AllowGaps = false) {
    if (SE && Len1) {
      const SCEV *Ptr1 = SE->getSCEV(Buf1);
      const SCEV *Ptr2 = SE->getSCEV(Buf2);
      const SCEV *Size1 = SE->getSCEV(Len1);

      if (!isa<SCEVCouldNotCompute>(Ptr1) && !isa<SCEVCouldNotCompute>(Ptr2) && !isa<SCEVCouldNotCompute>(Size1)) {
        const SCEV *ExtendedSize = SE->getTruncateOrZeroExtend(Size1, Ptr1->getType());
        const SCEV *ExpectedNext = SE->getAddExpr(Ptr1, ExtendedSize);
        if (ExpectedNext == Ptr2) return true; 
      }
    }

    APInt Off1(DL.getIndexTypeSizeInBits(Buf1->getType()), 0);
    const Value *Base1 = Buf1->stripAndAccumulateConstantOffsets(DL, Off1, true);
    APInt Off2(DL.getIndexTypeSizeInBits(Buf2->getType()), 0);
    const Value *Base2 = Buf2->stripAndAccumulateConstantOffsets(DL, Off2, true);

    if (Base1 && Base1 == Base2) {
      if (auto *CLen = dyn_cast_or_null<ConstantInt>(Len1)) {
        uint64_t End1 = Off1.getZExtValue() + CLen->getZExtValue();
        uint64_t Start2 = Off2.getZExtValue();
        if (End1 == Start2) return true;
      }
    }
    return false;
  }

  bool dependsOn(Value *V, Value *Target, int Depth = 0) {
    if (V == Target) return true;
    if (Depth > 4) return false; 
    if (auto *Inst = dyn_cast<Instruction>(V)) {
      for (Value *Op : Inst->operands()) {
	if (dependsOn(Op, Target, Depth + 1)) return true;
      }
    }
    return false;
  }

  bool isDeeplySafeFromIO(Function *F, SmallPtrSetImpl<Function*> &Visited) {
    // 1. If it's an external C library call, we can't see inside it. Unsafe.
    if (!F || F->isDeclaration()) return false; 
      
    // 2. Prevent infinite recursion on recursive functions (like in analyze.c!)
    if (!Visited.insert(F).second) return true; 

      
    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
	if (auto *LI = dyn_cast<LoadInst>(&I)) { if (LI->isVolatile()) return false; }
	else if (auto *SI = dyn_cast<StoreInst>(&I)) { if (SI->isVolatile()) return false; }
	else if (auto *MI = dyn_cast<MemIntrinsic>(&I)) { if (MI->isVolatile()) return false; }

	// Dealbreaker 2: Calls to unknown functions or known I/O
	if (auto *Call = dyn_cast<CallInst>(&I)) {
	  Function *SubCallee = Call->getCalledFunction();
                  
	  // If it calls a known I/O intrinsic (write, read, fsync, etc.), abort!
	  if (getIOArguments(Call, SubCallee).Type != IOArgs::NONE) return false;

	  // Do not treat sync calls as deelpy safe
	  if (SubCallee && SubCallee->hasName()) {
	    StringRef N = SubCallee->getName();
	    if (N == "fsync" || N == "fdatasync" || N == "msync" || N == "sync_file_range")
	      return false;
	  }
		  
	  // Recursively check the sub-function
	  if (!isDeeplySafeFromIO(SubCallee, Visited)) return false;
	}
      }
    }
    return true; // Clean bill of health! No I/O, no locks found.
  }

  bool isSafeToAddToBatch(const SmallVectorImpl<CallInst*> &Batch, CallInst *NewCall, AAResults &AA, const DataLayout &DL, ScalarEvolution &SE, DominatorTree &DT, PostDominatorTree &PDT) {
    if (Batch.empty()) return true;

    CallInst *LastCall = Batch.back();
    Function *LastCallee = LastCall->getCalledFunction();
    Function *NewCallee = NewCall->getCalledFunction();

    IOArgs FirstArgs = getIOArguments(Batch.front());
    IOArgs LastArgs = getIOArguments(LastCall, LastCallee);
    IOArgs NewArgs = getIOArguments(NewCall, NewCallee);

    if (NewArgs.Type == IOArgs::IO_SUBMIT || NewArgs.Type == IOArgs::AIO_WRITE || 
        NewArgs.Type == IOArgs::POSIX_PREADV || NewArgs.Type == IOArgs::POSIX_PWRITEV) {
      return false;
    }

    if (!FirstArgs.Buffer || !NewArgs.Buffer) return false;

    bool isReadBatch = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD || 
                        FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::MPI_READ_AT ||
                        FirstArgs.Type == IOArgs::CXX_READ);

    auto getPreciseLoc = [&](Value *Buf, Value *Len) {
      if (Len && isa<ConstantInt>(Len)) {
	return MemoryLocation(Buf, LocationSize::precise(cast<ConstantInt>(Len)->getZExtValue()));
      }
      return MemoryLocation(Buf, LocationSize::beforeOrAfterPointer());
    };

    if (!DT.dominates(LastCall, NewCall)) return false;

    if (isReadBatch) {
      if (NewArgs.Length) {
        if (auto *Inst = dyn_cast<Instruction>(NewArgs.Length)) {
	  if (!DT.dominates(Inst, Batch.front())) return false;
        }
      }
    }
   

    if (LastCallee != NewCallee) return false;

    Value *BaseFirst = getBaseFD(FirstArgs.Target);
    Value *BaseNew = getBaseFD(NewArgs.Target);
    if (!BaseFirst || !BaseNew || BaseFirst != BaseNew) return false;

    if (NewArgs.Type == IOArgs::SPLICE || NewArgs.Type == IOArgs::SENDFILE) {
      if (FirstArgs.Buffer != NewArgs.Buffer) return false;
    }

    if (NewArgs.Type == IOArgs::POSIX_PREAD || NewArgs.Type == IOArgs::POSIX_PWRITE) {
      Value *LastOffset = LastCall->getArgOperand(3);
      Value *NewOffset = NewCall->getArgOperand(3);
      Value *LastLen = LastArgs.Length;
      bool isContiguous = false;
    
      if (LastLen && SE.isSCEVable(LastOffset->getType()) && SE.isSCEVable(NewOffset->getType()) && SE.isSCEVable(LastLen->getType())) {
        const SCEV *SLast = SE.getSCEV(LastOffset);
        const SCEV *SNew = SE.getSCEV(NewOffset);
        const SCEV *SLen = SE.getSCEV(LastLen);
        if (!isa<SCEVCouldNotCompute>(SLast) && !isa<SCEVCouldNotCompute>(SNew) && !isa<SCEVCouldNotCompute>(SLen)) {
          const SCEV *ExpectedNext = SE.getAddExpr(SLast, SE.getTruncateOrZeroExtend(SLen, SLast->getType()));
          if (ExpectedNext == SNew) isContiguous = true;
        }
      }
      if (!isContiguous) return false;

    } else if (NewArgs.Type == IOArgs::MPI_READ_AT || NewArgs.Type == IOArgs::MPI_WRITE_AT) {
      if (LastCall->getArgOperand(5) != NewCall->getArgOperand(5)) return false;
      if (LastCall->getArgOperand(4) != NewCall->getArgOperand(4)) return false; 
      Value *LastOffset = LastCall->getArgOperand(1);
      Value *NewOffset = NewCall->getArgOperand(1);
      Value *LastCount = LastArgs.Length;
      bool isContiguous = false;

      if (LastCount && SE.isSCEVable(LastOffset->getType()) && SE.isSCEVable(NewOffset->getType())) {
        const SCEV *SLast = SE.getSCEV(LastOffset);
        const SCEV *SNew = SE.getSCEV(NewOffset);
        const SCEV *SCount = SE.getTruncateOrZeroExtend(SE.getSCEV(LastCount), SLast->getType());
        if (!isa<SCEVCouldNotCompute>(SLast) && !isa<SCEVCouldNotCompute>(SNew)) {
          const SCEV *ExpectedNext = SE.getAddExpr(SLast, SCount);
          if (ExpectedNext == SNew) isContiguous = true;
        }
      }
      if (!isContiguous) return false;
    } 

    BasicBlock *BB1 = LastCall->getParent();
    BasicBlock *BB2 = NewCall->getParent();

    if (BB1 != BB2) {
      if (!PDT.dominates(BB2, BB1)) {
	if (isReadBatch) return false;
          
	auto *Term = BB1->getTerminator();
	if (auto *Br = dyn_cast<BranchInst>(Term)) {
	  if (!Br->isConditional() || !dependsOn(Br->getCondition(), LastCall)) return false;
	} else {
	  return false;
	}
      }
    }
    
    LoadInst *Load1 = dyn_cast<LoadInst>(FirstArgs.Target);

    auto checkHazard = [&](Instruction *Inst) -> bool {
      if (auto *CI = dyn_cast<CallInst>(Inst)) {
	Function *Callee = CI->getCalledFunction();
          
	// Is it another I/O call? (Definite hazard, break batch)
	if (getIOArguments(CI, Callee).Type != IOArgs::NONE) return true;

	if (Callee) {
	  // Whitelist purely observational intrinsics
	  if (Callee->isIntrinsic()) {
	    Intrinsic::ID ID = Callee->getIntrinsicID();
	    // Do not skip lifetime_start/end. Moving I/O past them causes Use-After-Scope bugs.
	    // AA will handle lifetime markers correctly below.
	    if (ID == Intrinsic::dbg_value || ID == Intrinsic::dbg_declare || 
		ID == Intrinsic::dbg_label || ID == Intrinsic::assume) {
	      return false; // Safe to ignore!
	    }
	  }
	}

	if (Callee && Callee->hasName()) {
	  StringRef Name = Callee->getName();
	  // Known-safe functions that only read memory or do pure math
	  if (Name == "strlen" || Name == "strnlen" || Name == "strcmp" || 
              Name == "htons" || Name == "htonl" || Name == "ntohs" || Name == "ntohl" ||
              Name == "bswap_32" || Name == "bswap_64") {
	    return false; // Safely bypass the batch break!
	  }
	}

	if (!CI->onlyReadsMemory() && !CI->doesNotAccessMemory()) {
	  if (!CI->onlyAccessesArgMemory()) {
                  
	    // NEW: Before we panic and break the batch, let's look inside the function!
	    if (Callee && !Callee->isDeclaration()) {
	      SmallPtrSet<Function*, 8> Visited;
                      
	      if (isDeeplySafeFromIO(Callee, Visited)) {
		// SUCCESS! The function mutates global state (like setting 'errno'), 
		// but we mathematically proved it doesn't do I/O or acquire locks.
		// It is safe to let it fall through to the Alias Analysis checks below!
		return false; 
	      }
	    }

	    // If we get here, it's either an external black box or it contains real hazards.
	    StringRef BadFuncName = Callee ? Callee->getName() : "indirect_call";
	    logMessage("[IOOpt-Debug] Batch Break: Opaque function '" + BadFuncName + "' may interleave I/O or mutate global state.");
	    return true; 
	  }
	}
      }

      // Alias Analysis (AA) Checks
      if (FirstArgs.Type == IOArgs::SPLICE || FirstArgs.Type == IOArgs::SENDFILE) return false;

      if (Inst->mayReadOrWriteMemory()) {
	if (isReadBatch) {
	  if (NewArgs.Buffer->getType()->isPointerTy()) {
	    MemoryLocation NewLoc = getPreciseLoc(NewArgs.Buffer, NewArgs.Length);
	    // For reads: Check if the intervening instruction Reads or Writes our destination
	    if (isModOrRefSet(AA.getModRefInfo(Inst, NewLoc))) {
	      logMessage("[IOOpt-Debug] Batch Break: RAW/WAW dependency on new read buffer.");
	      return true;
	    }
	  }
	} else {
	  if (Inst->mayWriteToMemory()) {
	    for (CallInst *BC : Batch) {
	      IOArgs BArgs = getIOArguments(BC);
	      if (!BArgs.Buffer || !BArgs.Buffer->getType()->isPointerTy()) continue;
	      MemoryLocation BLoc = getPreciseLoc(BArgs.Buffer, BArgs.Length);
	      // For writes: Only check if the intervening instruction Modifies our source buffer
	      if (isModSet(AA.getModRefInfo(Inst, BLoc))) {
		logMessage("[IOOpt-Debug] Batch Break: WAR dependency on batched write buffer.");
		return true; 
	      }
	    }
	  }
	}

	if (FirstArgs.Target->getType()->isPointerTy()) {
	  MemoryLocation TargetLoc(FirstArgs.Target, LocationSize::beforeOrAfterPointer());
	  if (isModSet(AA.getModRefInfo(Inst, TargetLoc))) return true;
	}
	if (Load1 && Load1->getPointerOperand()->getType()->isPointerTy()) {
	  MemoryLocation FdLoc(Load1->getPointerOperand(), LocationSize::beforeOrAfterPointer());
	  if (isModSet(AA.getModRefInfo(Inst, FdLoc))) return true;
	}
      }
      return false; 
    };

    for (Instruction *I = LastCall->getNextNode(); I != nullptr; I = I->getNextNode()) {
      if (I == NewCall) return true; 
      if (checkHazard(I)) return false;
    }

    SmallPtrSet<BasicBlock*, 8> Visited;
    SmallVector<BasicBlock*, 16> Worklist; 
    for (BasicBlock *Succ : successors(BB1)) {
      if (Succ != BB2) {
        Worklist.push_back(Succ);
        Visited.insert(Succ);
      }
    }

    while (!Worklist.empty()) {
      BasicBlock *CurrBB = Worklist.pop_back_val();
      for (Instruction &I : *CurrBB) {
        if (checkHazard(&I)) return false;
      }
      for (BasicBlock *Succ : successors(CurrBB)) {
        if (!DT.dominates(BB1, Succ)) return false; 
        if (Succ != BB2 && Visited.insert(Succ).second) {
          Worklist.push_back(Succ);
        }
      }
    }

    for (Instruction &I : *BB2) {
      if (&I == NewCall) break; 
      if (checkHazard(&I)) return false;
    }

    return true; 
  }

  enum class IOPattern { Contiguous, Strided, ShadowBuffer, DynamicShadowBuffer, Vectored, Unprofitable };

  IOPattern classifyBatch(const SmallVectorImpl<CallInst*> &Batch, const DataLayout &DL, 
                          uint64_t &OutTotalRange, ScalarEvolution *SE) {
    if (Batch.size() < 2) return IOPattern::Unprofitable;

    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isReadBatch = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD || FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::CXX_READ);
    
    if (FirstArgs.Type == IOArgs::SPLICE || FirstArgs.Type == IOArgs::SENDFILE) return IOPattern::Contiguous; 

    bool StrictPhysical = true;
    for (size_t i = 0; i < Batch.size() - 1; ++i) {
      if (!checkAdjacency(getIOArguments(Batch[i]).Buffer, getIOArguments(Batch[i]).Length, 
                          getIOArguments(Batch[i+1]).Buffer, DL, SE, false)) {
        StrictPhysical = false;
        break;
      }
    }
    if (StrictPhysical) return IOPattern::Contiguous;
    
    bool isWriteBatch = (FirstArgs.Type == IOArgs::POSIX_WRITE || FirstArgs.Type == IOArgs::POSIX_PWRITE || 
                         FirstArgs.Type == IOArgs::MPI_WRITE_AT || FirstArgs.Type == IOArgs::C_FWRITE || 
                         FirstArgs.Type == IOArgs::CXX_WRITE);

    if (isWriteBatch) {
      bool isConstantTinySize = true;
      uint64_t ElemSize = 0;
      if (auto *CSize = dyn_cast_or_null<ConstantInt>(FirstArgs.Length)) {
	ElemSize = CSize->getZExtValue();
	if (ElemSize != 1 && ElemSize != 2 && ElemSize != 4 && ElemSize != 8) {
	  isConstantTinySize = false;
	} else {
	  for (CallInst *C : Batch) {
	    auto *CS = dyn_cast_or_null<ConstantInt>(getIOArguments(C).Length);
	    if (!CS || CS->getZExtValue() != ElemSize) {
	      isConstantTinySize = false;
	      break;
	    }
	  }
	}
      } else {
	isConstantTinySize = false;
      }

      if (isConstantTinySize && Batch.size() >= 2 && Batch.size() <= 64) {
	OutTotalRange = ElemSize; 
	return IOPattern::Strided;
      }
    }
    
    // Reads are high latency; convert even 2 reads to readv.
    // Writes are buffered; require Config.BatchThreshold for writev.
    if (isReadBatch || Batch.size() >= Config.BatchThreshold) {
      if (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::POSIX_WRITE || 
          FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::POSIX_PWRITE) {
        return IOPattern::Vectored;
      }
    }

    if (isWriteBatch) { 
      uint64_t TotalConstSize = 0;
      bool AllSizesConstant = true;
        
      for (CallInst *C : Batch) {
        IOArgs CArgs = getIOArguments(C);
        if (CArgs.Length && isa<ConstantInt>(CArgs.Length)) {
          TotalConstSize += cast<ConstantInt>(CArgs.Length)->getZExtValue();
        } else {
          AllSizesConstant = false;
          break;
        }
      }
        
      if (AllSizesConstant && TotalConstSize > 0 && TotalConstSize <= Config.ShadowBufferSize) {
        OutTotalRange = TotalConstSize;
        return IOPattern::ShadowBuffer;
      }
      
      if (Batch.size() >= Config.BatchThreshold) {
	return IOPattern::DynamicShadowBuffer;
      }
    }

    return IOPattern::Unprofitable;
  }

  bool flushBatch(SmallVectorImpl<CallInst*> &Batch, Module *M, ScalarEvolution &SE, DominatorTree *DT = nullptr) { 
    if (Batch.empty()) return false;

    const DataLayout &DL = M->getDataLayout();
    uint64_t TotalConstSize = 0;
    
    IOPattern Pattern = classifyBatch(Batch, DL, TotalConstSize, &SE);

    if (Pattern == IOPattern::Vectored && Batch.size() > Config.MaxIov) {
      // Split into chunks of MaxIov and flush chunk-by-chunk.
      // (Minimal approach: return false and leave scalar calls, but chunking is better.)
    }
    
    if (Pattern == IOPattern::Unprofitable) {
      Batch.clear();
      return false; 
    }

    //
    // Even if run-level batching accidentally forms an oversized batch, never emit a
    // single readv/writev/preadv/pwritev with iovcnt > MaxIov (typically 1024 on Linux).
    //
    // We conservatively split the batch into chunks and flush them in order.
    if (Pattern == IOPattern::Vectored) {
      unsigned Limit = Config.MaxIov;
      if (Limit == 0) Limit = 1024;
      if (Batch.size() > Limit) {
	bool AnyChanged = false;
	
	// Process in-order chunks of at most Limit calls.
	size_t I = 0;
	while (I < Batch.size()) {
	  size_t End = I + (size_t)Limit;
	  if (End > Batch.size()) End = Batch.size();
	  
	  SmallVector<CallInst*, 1024> Chunk;
	  Chunk.append(Batch.begin() + I, Batch.begin() + End);
	  
	  // Recursively flush each chunk. Each chunk is <= Limit, so this will terminate.
	  AnyChanged |= flushBatch(Chunk, M, SE, DT);
	  
	  I = End;
	}
	
	// We have consumed the logical batch; clear the original vector so callers
	// don't try to reuse it.
	Batch.clear();
	return AnyChanged;
      }
    }
    
    

    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isRead = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD || FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::MPI_READ_AT || FirstArgs.Type == IOArgs::CXX_READ);
    bool isExplicit = (FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::POSIX_PWRITE);

    Instruction *InsertPt = isRead ? Batch.front() : Batch.back();
    IRBuilder<> InsertBuilder(InsertPt);

    Value *TotalDynLen = InsertBuilder.getIntN(FirstArgs.Length->getType()->getIntegerBitWidth(), 0);
    for (CallInst *C : Batch) {
      Value *L = getIOArguments(C).Length;
      if (L && L->getType() != TotalDynLen->getType()) L = InsertBuilder.CreateZExtOrTrunc(L, TotalDynLen->getType());
      if (L) TotalDynLen = InsertBuilder.CreateAdd(TotalDynLen, L, "dyn.len.add"); 
    }
 
    CallInst *MergedCall = nullptr;

    auto buildArgs = [&](Value *DataBuf) -> SmallVector<Value*, 8> {
      SmallVector<Value*, 8> NewArgs; 
      Type *ExpectedBufTy = InsertBuilder.getPtrTy();
      if (DataBuf && DataBuf->getType() != ExpectedBufTy && DataBuf->getType()->isPointerTy()) {
	DataBuf = InsertBuilder.CreatePointerBitCastOrAddrSpaceCast(DataBuf, ExpectedBufTy);
      }

      if (FirstArgs.Type == IOArgs::MPI_WRITE_AT || FirstArgs.Type == IOArgs::MPI_READ_AT) {
	NewArgs = { Batch[0]->getArgOperand(0), Batch[0]->getArgOperand(1), DataBuf, TotalDynLen, Batch[0]->getArgOperand(4), Batch[0]->getArgOperand(5) };
      } else if (FirstArgs.Type == IOArgs::C_FWRITE || FirstArgs.Type == IOArgs::C_FREAD) {
	Value *SizeOne = InsertBuilder.getIntN(TotalDynLen->getType()->getIntegerBitWidth(), 1);
	NewArgs = {DataBuf, SizeOne, TotalDynLen, FirstArgs.Target};
      } else if (FirstArgs.Type == IOArgs::SPLICE) {
	NewArgs = {Batch[0]->getArgOperand(0), Batch[0]->getArgOperand(1), Batch[0]->getArgOperand(2), Batch[0]->getArgOperand(3), TotalDynLen, Batch[0]->getArgOperand(5)};
      } else if (FirstArgs.Type == IOArgs::SENDFILE) {
	NewArgs = {Batch[0]->getArgOperand(0), Batch[0]->getArgOperand(1), Batch[0]->getArgOperand(2), TotalDynLen};
      } else if (isExplicit) {
	NewArgs = {FirstArgs.Target, DataBuf, TotalDynLen, Batch[0]->getArgOperand(3)};
      } else {
	NewArgs = {FirstArgs.Target, DataBuf, TotalDynLen};
      }
      return NewArgs;
    };

    switch (Pattern) {
    case IOPattern::Contiguous: {
      MergedCall = InsertBuilder.CreateCall(Batch[0]->getCalledFunction(), buildArgs(FirstArgs.Buffer));
      if (FirstArgs.Type == IOArgs::SPLICE || FirstArgs.Type == IOArgs::SENDFILE) {
	NumZeroCopy++;
	logMessage("[IOOpt] SUCCESS: N-Way zero-copy kernel transfer merged " + Twine(Batch.size()) + " calls.");
      } else {
	logMessage("[IOOpt] SUCCESS: N-Way contiguous batch merged " + Twine(Batch.size()) + " calls.");
      }
      NumBatchesMerged++; 
      break;
    }

    case IOPattern::Strided: {
      unsigned ElementBytes = TotalConstSize; 
      unsigned NumElements = Batch.size();
      Type *ElementTy = InsertBuilder.getIntNTy(ElementBytes * 8); 
      auto *VecTy = FixedVectorType::get(ElementTy, NumElements);
      Value *GatherVec = PoisonValue::get(VecTy);
      for (unsigned i = 0; i < NumElements; ++i) {
        IOArgs Args = getIOArguments(Batch[i]);
        Value *SafeBufPtr = Args.Buffer;
        if (SafeBufPtr->getType() != InsertBuilder.getPtrTy() && SafeBufPtr->getType()->isPointerTy()) {
	  SafeBufPtr = InsertBuilder.CreatePointerBitCastOrAddrSpaceCast(SafeBufPtr, InsertBuilder.getPtrTy());
        }
        LoadInst *LoadedVal = InsertBuilder.CreateLoad(ElementTy, SafeBufPtr, "strided.load");
        GatherVec = InsertBuilder.CreateInsertElement(GatherVec, LoadedVal, InsertBuilder.getInt32(i), "gather.insert");
      }
      Function *F = Batch.back()->getFunction();
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());
      AllocaInst *ContiguousBuf = EntryBuilder.CreateAlloca(VecTy, nullptr, "simd.shadow.buf");
      ContiguousBuf->setAlignment(Align(64));
      InsertBuilder.CreateStore(GatherVec, ContiguousBuf);
      Value *BufCast = InsertBuilder.CreatePointerCast(ContiguousBuf, InsertBuilder.getPtrTy());
      
      MergedCall = InsertBuilder.CreateCall(Batch[0]->getCalledFunction(), buildArgs(BufCast));
      NumBatchesMerged++; 
      logMessage("[IOOpt] SUCCESS: N-Way strided SIMD batch created for " + Twine(Batch.size()) + " calls.");
      break;
    }

    case IOPattern::ShadowBuffer: {
      Function *F = Batch.back()->getFunction();
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());

      Type *Int8Ty = InsertBuilder.getInt8Ty();
      ArrayType *ShadowArrTy = ArrayType::get(Int8Ty, TotalConstSize);
      AllocaInst *ShadowBuf = EntryBuilder.CreateAlloca(ShadowArrTy, nullptr, "shadow.buf");
      ShadowBuf->setAlignment(Align(64)); 

      uint64_t CurrentOffset = 0;
      for (size_t i = 0; i < Batch.size(); ++i) {
        CallInst *C = Batch[i];
        IOArgs Args = getIOArguments(C);
        IRBuilder<> CallBuilder(C);
        Value *DestPtr = CallBuilder.CreateInBoundsGEP(ShadowArrTy, ShadowBuf, {CallBuilder.getInt32(0), CallBuilder.getInt32(CurrentOffset)});
        CallBuilder.CreateMemCpy(DestPtr, Align(1), Args.Buffer, Align(1), Args.Length);
        if (auto *ConstLen = dyn_cast_or_null<ConstantInt>(Args.Length)) CurrentOffset += ConstLen->getZExtValue();
      }

      Value *BufPtr = InsertBuilder.CreatePointerCast(ShadowBuf, InsertBuilder.getPtrTy());
      MergedCall = InsertBuilder.CreateCall(Batch[0]->getCalledFunction(), buildArgs(BufPtr));
      NumBatchesMerged++; 
      logMessage("[IOOpt] SUCCESS: N-Way static ShadowBuffer merged " + Twine(Batch.size()) + " calls (" + Twine(TotalConstSize) + " bytes).");
      break;
    }

    case IOPattern::DynamicShadowBuffer: {
      Type *SizeTy = DL.getIntPtrType(M->getContext());
      Type *Int8Ty = InsertBuilder.getInt8Ty();
      PointerType *PtrTy = InsertBuilder.getPtrTy(); // opaque ptr
      Type *Int32Ty = InsertBuilder.getInt32Ty();
      Type *VoidTy  = InsertBuilder.getVoidTy();

      // int posix_memalign(void **memptr, size_t alignment, size_t size)
      // In opaque pointer mode, both void* and void** are represented as 'ptr' at the type level;
      // correctness comes from passing the *address of a pointer slot* for memptr.
      FunctionType *PosixMemalignTy =
	FunctionType::get(Int32Ty, {PtrTy, SizeTy, SizeTy}, false);
      FunctionCallee MemAlignFunc =
	M->getOrInsertFunction("posix_memalign", PosixMemalignTy);

      FunctionType *FreeTy = FunctionType::get(VoidTy, {PtrTy}, false);
      FunctionCallee FreeFunc = M->getOrInsertFunction("free", FreeTy);

      // int dprintf(int fd, const char *fmt, ...);
      FunctionType *DprintfTy = FunctionType::get(Int32Ty, {Int32Ty, PtrTy}, true);
      FunctionCallee DprintfFn = M->getOrInsertFunction("dprintf", DprintfTy);

      // void abort(void);
      FunctionType *AbortTy = FunctionType::get(VoidTy, {}, false);
      FunctionCallee AbortFn = M->getOrInsertFunction("abort", AbortTy);

      Function *F = Batch.back()->getFunction();

      // Entry block allocas so TrapBB can load diagnostics.
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());

      // Slot to receive heap pointer from posix_memalign (holds a 'ptr' value).
      AllocaInst *HeapBufPtr = EntryBuilder.CreateAlloca(PtrTy, nullptr, "dyn.shadow.ptr");
      HeapBufPtr->setAlignment(Align(alignof(void *)));

      // Diagnostic slots (rc and size)
      AllocaInst *RCSlot = EntryBuilder.CreateAlloca(Int32Ty, nullptr, "ioopt.pmem.rc");
      RCSlot->setAlignment(Align(4));

      AllocaInst *SizeSlot = EntryBuilder.CreateAlloca(SizeTy, nullptr, "ioopt.pmem.size");
      SizeSlot->setAlignment(Align(alignof(size_t)));

      // Split the block at InsertPt so we can branch to a fail-fast block before memcpy/write.
      BasicBlock *OrigBB = InsertPt->getParent();
      BasicBlock *ContBB = OrigBB->splitBasicBlock(InsertPt, "ioopt.dynshadow.cont");

      // Create trap/diagnostic block (placed before ContBB for nicer layout)
      BasicBlock *TrapBB = BasicBlock::Create(M->getContext(), "ioopt.dynshadow.fail", F, ContBB);

      // Insert allocation + check in OrigBB, right before its terminator
      Instruction *OrigTerm = OrigBB->getTerminator();
      IRBuilder<> PreBuilder(OrigTerm);

      Value *MallocSize = PreBuilder.CreateZExtOrTrunc(TotalDynLen, SizeTy);
      Value *AlignVal = ConstantInt::get(SizeTy, 64);

      // posix_memalign(&ptr, align, size)
      Value *RC = PreBuilder.CreateCall(MemAlignFunc, {HeapBufPtr, AlignVal, MallocSize}, "pmem.rc");
      PreBuilder.CreateStore(RC, RCSlot);
      PreBuilder.CreateStore(MallocSize, SizeSlot);

      Value *HeapBuf = PreBuilder.CreateLoad(PtrTy, HeapBufPtr, "dyn.shadow.buf");

      // ok = (rc == 0) && (buf != null)
      Value *OkRC = PreBuilder.CreateICmpEQ(RC, ConstantInt::get(Int32Ty, 0), "pmem.ok.rc");
      Value *NonNull =
	PreBuilder.CreateICmpNE(HeapBuf, ConstantPointerNull::get(PtrTy), "pmem.nonnull");
      Value *Ok = PreBuilder.CreateAnd(OkRC, NonNull, "pmem.ok");

      // Replace unconditional branch inserted by splitBasicBlock with conditional branch
      OrigTerm->eraseFromParent();
      BranchInst::Create(ContBB, TrapBB, Ok, OrigBB);

      // ---- TrapBB: print diagnostics and abort ----
      IRBuilder<> TrapBuilder(TrapBB);
      Value *RCVal = TrapBuilder.CreateLoad(Int32Ty, RCSlot, "ioopt.pmem.rc.val");
      Value *SizeVal = TrapBuilder.CreateLoad(SizeTy, SizeSlot, "ioopt.pmem.size.val");

      Value *Fmt = TrapBuilder.CreateGlobalString(
						  "IOOpt: posix_memalign failed (rc=%d, size=%zu)\\n",
						  "ioopt.pmem.fmt");

      // dprintf(2, fmt, rc, size)
      TrapBuilder.CreateCall(DprintfFn, {TrapBuilder.getInt32(2), Fmt, RCVal, SizeVal});
      TrapBuilder.CreateCall(AbortFn);
      TrapBuilder.CreateUnreachable();

      // ---- ContBB: build the contiguous heap buffer, emit merged call, free ----
      IRBuilder<> ContBuilder(&*ContBB->getFirstInsertionPt());

      Value *CurrentOffset = ConstantInt::get(SizeTy, 0);
      for (size_t i = 0; i < Batch.size(); ++i) {
	CallInst *C = Batch[i];
	IOArgs Args = getIOArguments(C);

	Value *Len = ContBuilder.CreateZExtOrTrunc(Args.Length, SizeTy);
	Value *DestPtr =
	  ContBuilder.CreateInBoundsGEP(Int8Ty, HeapBuf, CurrentOffset, "dyn.dest");

	ContBuilder.CreateMemCpy(DestPtr, Align(1), Args.Buffer, Align(1), Len);
	CurrentOffset = ContBuilder.CreateAdd(CurrentOffset, Len, "dyn.offset");
      }

      // Emit merged call using contiguous heap buffer
      MergedCall = ContBuilder.CreateCall(Batch[0]->getCalledFunction(), buildArgs(HeapBuf));

      // Free heap buffer
      ContBuilder.CreateCall(FreeFunc, {HeapBuf});

      NumBatchesMerged++;
      logMessage("[IOOpt] SUCCESS: N-Way dynamic ShadowBuffer merged " + Twine(Batch.size()) + " calls.");
      break;
    }


    case IOPattern::Vectored: {
      Type *Int32Ty = InsertBuilder.getInt32Ty();
      Type *PtrTy = InsertBuilder.getPtrTy();
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
      IovArray->setAlignment(Align(8));
      
      for (size_t i = 0; i < Batch.size(); ++i) {
        IOArgs Args = getIOArguments(Batch[i]);
        Value *IovPtr = InsertBuilder.CreateInBoundsGEP(IovArrayTy, IovArray, {InsertBuilder.getInt32(0), InsertBuilder.getInt32(i)});
        
        Value *SafeBufPtr = Args.Buffer;

        if (DT && isa<Instruction>(SafeBufPtr) && !DT->dominates(cast<Instruction>(SafeBufPtr), InsertPt)) {
	  SCEVExpander Expander(SE, DL, "io.vectored.expander");
	  SafeBufPtr = Expander.expandCodeFor(SE.getSCEV(SafeBufPtr), SafeBufPtr->getType(), InsertPt);
        }

        if (SafeBufPtr->getType() != PtrTy && SafeBufPtr->getType()->isPointerTy()) {
	  SafeBufPtr = InsertBuilder.CreatePointerBitCastOrAddrSpaceCast(SafeBufPtr, PtrTy);
        }       
 
        InsertBuilder.CreateStore(SafeBufPtr, InsertBuilder.CreateStructGEP(IovecTy, IovPtr, 0));
        InsertBuilder.CreateStore(InsertBuilder.CreateIntCast(Args.Length, SizeTy, false), InsertBuilder.CreateStructGEP(IovecTy, IovPtr, 1));
      }
            
      Value *IovBasePtr = InsertBuilder.CreateInBoundsGEP(IovArrayTy, IovArray, {InsertBuilder.getInt32(0), InsertBuilder.getInt32(0)}, "iovec.base.ptr");
      Value *Fd = InsertBuilder.CreateIntCast(FirstArgs.Target, Int32Ty, false);
      if (isExplicit) {
        MergedCall = InsertBuilder.CreateCall(VecFunc, {Fd, IovBasePtr, InsertBuilder.getInt32(Batch.size()), Batch[0]->getArgOperand(3)});
      } else {
        MergedCall = InsertBuilder.CreateCall(VecFunc, {Fd, IovBasePtr, InsertBuilder.getInt32(Batch.size())});
      }
      NumBatchesMerged++; 
      logMessage("[IOOpt] SUCCESS: N-Way converted " + Twine(Batch.size()) + " " + (isRead ? "reads" : "writes") + " to " + FuncName + "!");
      break;
    }
    default: break;
    }

    IRBuilder<> RetBuilder(MergedCall->getNextNode());
    
    for (size_t i = 0; i < Batch.size(); ++i) {
      CallInst *C = Batch[i];
      if (C->use_empty()) {
	C->eraseFromParent();
	continue;
      }

      IOArgs CArgs = getIOArguments(C);
      Value *Rep = nullptr;

      if (CArgs.Type == IOArgs::CXX_WRITE) {
	Rep = C->getArgOperand(0); 
      } else if (CArgs.Type == IOArgs::MPI_WRITE_AT || CArgs.Type == IOArgs::MPI_READ_AT) {
	Rep = RetBuilder.getInt32(0); 
      } else {
	Value *ExpectedLen = CArgs.Length;
	if (CArgs.Type == IOArgs::C_FWRITE || CArgs.Type == IOArgs::C_FREAD) ExpectedLen = C->getArgOperand(2);

	if (!isRead && i != Batch.size() - 1) {
	  Rep = ExpectedLen; 
	} else {
	  Value *RealRet = MergedCall;
	  if (RealRet->getType() != ExpectedLen->getType()) RealRet = RetBuilder.CreateIntCast(RealRet, ExpectedLen->getType(), true);
              
	  if (FirstArgs.Type == IOArgs::POSIX_WRITE || FirstArgs.Type == IOArgs::POSIX_READ || 
	      FirstArgs.Type == IOArgs::POSIX_PWRITE || FirstArgs.Type == IOArgs::POSIX_PREAD ||
	      FirstArgs.Type == IOArgs::SPLICE || FirstArgs.Type == IOArgs::SENDFILE) {
	    Value *Zero = RetBuilder.getIntN(RealRet->getType()->getIntegerBitWidth(), 0);
	    Value *IsErr = RetBuilder.CreateICmpSLT(RealRet, Zero);
	    Rep = RetBuilder.CreateSelect(IsErr, RealRet, ExpectedLen, "spoofed.posix.ret");
	  } else {
	    Value *TotalDynCast = RetBuilder.CreateIntCast(TotalDynLen, RealRet->getType(), false);
	    Value *IsPerfect = RetBuilder.CreateICmpEQ(RealRet, TotalDynCast);
	    Value *Zero = RetBuilder.getIntN(ExpectedLen->getType()->getIntegerBitWidth(), 0);
	    Rep = RetBuilder.CreateSelect(IsPerfect, ExpectedLen, Zero, "spoofed.cstream.ret");
	  }
	}
	if (C->getType() != Rep->getType()) Rep = RetBuilder.CreateIntCast(Rep, C->getType(), false);
      }
      C->replaceAllUsesWith(Rep);
      C->eraseFromParent();
    }
    
    Batch.clear();
    return true;
  }

  struct IOOptimisationPass : public PassInfoMixin<IOOptimisationPass> {
    /// Return the pointer operand for common memory-writing instructions.
    /// If we cannot confidently extract a destination pointer, return nullptr
    /// (conservatively blocks hoisting when aliasing is possible).
    static Value *getMemoryWritePtr(Instruction &I) {
      if (auto *SI = dyn_cast<StoreInst>(&I)) return SI->getPointerOperand();
      if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) return RMW->getPointerOperand();
      if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) return CX->getPointerOperand();
      if (auto *MT = dyn_cast<MemTransferInst>(&I)) return MT->getDest();
      if (auto *MS = dyn_cast<MemSetInst>(&I)) return MS->getDest();
      return nullptr;
    }

    /// Conservative proof that it is safe to hoist a loop-contained POSIX-style
    /// read/write out of the loop, without changing the *data* written/read by
    /// each iteration.
    ///
    /// This is designed primarily for hoisting WRITE-like calls that read from
    /// the user buffer; it blocks in the presence of ambiguous clobbers.
    static bool isSafeToHoistLoopIOCall(
					CallInst *Call,
					const IOArgs &Args,
					Loop *L,
					ScalarEvolution &SE,
					const DataLayout &DL,
					AAResults &AA,
					DominatorTree &DT,
					MemorySSA &MSSA) {

      // Only attempt on calls where we can reason about buffer and length.
      if (!Args.Buffer || !Args.Length) return false;
      if (!isa<ConstantInt>(Args.Length)) return false;

      auto *LenC = cast<ConstantInt>(Args.Length);
      uint64_t ElemLen = LenC->getZExtValue();
      if (ElemLen == 0) return false;

      // Require a computable constant trip count so we can precisely bound the range.
      const SCEV *BackedgeCount = SE.getBackedgeTakenCount(L);
      if (isa<SCEVCouldNotCompute>(BackedgeCount)) return false;
      auto *BEC = dyn_cast<SCEVConstant>(BackedgeCount);
      if (!BEC) return false;

      uint64_t Trips = BEC->getAPInt().getZExtValue() + 1;
      if (Trips == 0) return false;

      // Avoid overflow when forming a precise LocationSize.
      if (Trips > (std::numeric_limits<uint64_t>::max() / ElemLen)) return false;
      uint64_t TotalLen = Trips * ElemLen;

      // Buffer must be an addrec in *this* loop with step == element size.
      const SCEV *BufS = SE.getSCEV(Args.Buffer);
      auto *BufAR = dyn_cast<SCEVAddRecExpr>(BufS);
      if (!BufAR || BufAR->getLoop() != L) return false;

      const SCEV *BufStep = SE.getTruncateOrZeroExtend(BufAR->getStepRecurrence(SE),
                                                       DL.getIntPtrType(Call->getContext()));
      const SCEV *ElemS  = SE.getTruncateOrZeroExtend(SE.getSCEV(Args.Length),
						      DL.getIntPtrType(Call->getContext()));
      if (!SE.isKnownNonNegative(BufStep)) return false;
      if (!SE.isKnownPredicate(ICmpInst::ICMP_EQ, BufStep, ElemS)) return false;

      const SCEV *Start = BufAR->getStart();
      auto *U = dyn_cast<SCEVUnknown>(Start);
      if (!U) return false;
      Value *BasePtr = U->getValue();
      if (!BasePtr || !BasePtr->getType()->isPointerTy()) return false;

      // Precise full-range location that would be written/read by hoisting.
      MemoryLocation FullRange(BasePtr, LocationSize::precise(TotalLen));

      // MemorySSA-style scan: only consider MemoryDefs (actual clobbers).
      // Any ambiguous clobber of FullRange that can occur after the per-iteration
      // I/O call would make hoisting unsafe.
      for (BasicBlock *BB : L->blocks()) {
        for (Instruction &I : *BB) {
          if (&I == Call) continue;
          if (!I.mayWriteToMemory()) continue;

          // If MemorySSA has no access for this instruction, treat conservatively.
          // (Most memory-writing instructions should have one.)
          MemoryAccess *MA = MSSA.getMemoryAccess(&I);
          if (!MA) return false;
          if (!isa<MemoryDef>(MA)) continue;

          // If it doesn't mod our full range, ignore it.
          if (!isModSet(AA.getModRefInfo(&I, FullRange))) continue;

          // It must dominate the I/O call, meaning it is guaranteed to execute
          // before the call on all paths (and, in the same block, appear earlier).
          if (!DT.dominates(&I, Call)) return false;

          // Further restrict: the write should be to the same per-iteration "slice family"
          // as the I/O buffer addrec. If we cannot prove this, block hoisting.
          Value *WPtr = getMemoryWritePtr(I);
          if (!WPtr) return false;

          const SCEV *WS = SE.getSCEV(WPtr);
          auto *WAR = dyn_cast<SCEVAddRecExpr>(WS);
          if (!WAR || WAR->getLoop() != L) return false;

          const SCEV *WStep = SE.getTruncateOrZeroExtend(WAR->getStepRecurrence(SE),
							 DL.getIntPtrType(Call->getContext()));
          if (!SE.isKnownNonNegative(WStep)) return false;
	  if (!SE.isKnownPredicate(ICmpInst::ICMP_EQ, WStep, BufStep)) return false;

          // Allow constant field offsets within the element: WAR.start = BufAR.start  k
          // with 0 <= k < ElemLen.
          const SCEV *StartDiff = SE.getMinusSCEV(WAR->getStart(), BufAR->getStart());
          auto *CD = dyn_cast<SCEVConstant>(StartDiff);
          if (!CD) return false;
          uint64_t Off = CD->getAPInt().getZExtValue();
          if (Off >= ElemLen) return false;
        }
      }

      return true;
    }


    bool optimiseLoopIO(Loop *L, ScalarEvolution &SE, const DataLayout &DL, LoopInfo &LI, DominatorTree &DT, AAResults &AA, MemorySSA &MSSA) {    
                              
      BasicBlock *Preheader = L->getLoopPreheader();
      BasicBlock *ExitBB = L->getExitBlock();
      if (!Preheader || !ExitBB) return false;

      if (!L->isLoopSimplifyForm() || !L->isLCSSAForm(DT)) {
	return false;
      }

      const SCEV *BackedgeCount = SE.getBackedgeTakenCount(L);
      if (isa<SCEVCouldNotCompute>(BackedgeCount)) return false;
      
      Type *IntPtrTy = DL.getIntPtrType(Preheader->getContext());
      const SCEV *TripCountSCEV = SE.getAddExpr(SE.getTruncateOrZeroExtend(BackedgeCount, IntPtrTy), SE.getOne(IntPtrTy));

      bool LoopChanged = false;
      Loop *HoistLoop = L;
      BasicBlock *HoistPreheader = HoistLoop->getLoopPreheader();
      BasicBlock *HoistExitBB = HoistLoop->getExitBlock();
      SCEVExpander Expander(SE, DL, "io.dyn.expander");

      for (BasicBlock *BB : L->blocks()) {
        for (Instruction &I : llvm::make_early_inc_range(*BB)) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            Function *CalleeF = Call->getCalledFunction();
            IOArgs Args = getIOArguments(Call, CalleeF);
            
            bool isWrite = (Args.Type == IOArgs::POSIX_WRITE || Args.Type == IOArgs::C_FWRITE || Args.Type == IOArgs::CXX_WRITE);
            bool isRead = (Args.Type == IOArgs::POSIX_READ || Args.Type == IOArgs::C_FREAD || Args.Type == IOArgs::CXX_READ);

            if (isWrite || isRead) {
	      // Apply strict safety check before hoisting attempt
	      if (!isSafeToHoistLoopIOCall(Call, Args, L, SE, DL, AA, DT, MSSA)) {
                continue;
              }

              bool hasSideEffects = false;
              for (BasicBlock *ScanBB : L->blocks()) {
		for (Instruction &ScanInst : *ScanBB) {
		  if (&ScanInst == Call) continue;
                      
		  if (Args.Target->getType()->isPointerTy() && ScanInst.mayWriteToMemory()) {
		    MemoryLocation TargetLoc(Args.Target, LocationSize::beforeOrAfterPointer());
		    if (isModSet(AA.getModRefInfo(&ScanInst, TargetLoc))) {
		      logMessage("[IOOpt-Debug] Loop Hoist Blocked: Loop contains aliased mutation of File Stream.");
		      hasSideEffects = true;
		      break;
		    }
		  }
                      
		  if (auto *ScanCall = dyn_cast<CallInst>(&ScanInst)) {
		    if (getIOArguments(ScanCall).Type != IOArgs::NONE || (!ScanCall->onlyReadsMemory() && !ScanCall->doesNotAccessMemory())) {
		      logMessage("[IOOpt-Debug] Loop Hoist Blocked: Opaque call or interleaved I/O would scramble temporal order.");
		      hasSideEffects = true;
		      break;
		    }
		  }
		}
		if (hasSideEffects) break;
              }
              if (hasSideEffects) continue;

              if (!Args.Length || !HoistLoop->isLoopInvariant(Args.Length)) continue;

              Value *ExtraArg = nullptr;
              if (Args.Type == IOArgs::C_FWRITE || Args.Type == IOArgs::C_FREAD) {
                ExtraArg = Call->getArgOperand(1);
                if (!HoistLoop->isLoopInvariant(ExtraArg)) continue; 
              }

              if (!HoistLoop->isLoopInvariant(Args.Target)) continue;

              const SCEV *ElementSizeSCEV = SE.getTruncateOrZeroExtend(SE.getSCEV(Args.Length), IntPtrTy);
              const SCEV *TotalBytesSCEV = SE.getMulExpr(ElementSizeSCEV, TripCountSCEV);

              const SCEV *PtrSCEV = SE.getSCEV(Args.Buffer);
              Value *BasePointer = nullptr;

              if (auto *AddRec = dyn_cast<SCEVAddRecExpr>(PtrSCEV)) {
                if (AddRec->getLoop() != L) continue;
                const SCEV *StepSCEV = SE.getTruncateOrZeroExtend(AddRec->getStepRecurrence(SE), IntPtrTy);
                
                if (auto *StepConst = dyn_cast<SCEVConstant>(StepSCEV)) {
		  if (StepConst->getValue()->isNegative()) continue; 
                }

                if (StepSCEV != ElementSizeSCEV) continue;
                if (!SE.isLoopInvariant(AddRec->getStart(), HoistLoop)) continue;
                BasePointer = Expander.expandCodeFor(AddRec->getStart(), Args.Buffer->getType(), HoistPreheader->getTerminator());
              }

              if (!BasePointer) continue;

              Instruction *InsertionPoint = isRead ? HoistPreheader->getTerminator() : &*HoistExitBB->getFirstInsertionPt();
              IRBuilder<> Builder(InsertionPoint);
              
              // 1. Safely expand using the exact type the SCEV was built with
              Value *TotalLenVal = Expander.expandCodeFor(TotalBytesSCEV, IntPtrTy, InsertionPoint);

              if (TotalLenVal->getType() != Args.Length->getType()) {
                TotalLenVal = Builder.CreateIntCast(TotalLenVal, Args.Length->getType(), false);
              }             
 
              SmallVector<Value *, 8> NewArgs; 
              if (Args.Type == IOArgs::C_FWRITE || Args.Type == IOArgs::C_FREAD) {
                NewArgs = {BasePointer, ExtraArg, TotalLenVal, Args.Target};
              } else {
                NewArgs = {Args.Target, BasePointer, TotalLenVal};
              }
              Builder.CreateCall(Call->getCalledFunction(), NewArgs);

              NumLoopsHoisted++;
              logMessage(isRead ? "[IOOpt] SUCCESS: Hoisted DYNAMIC READ to Preheader!" : "[IOOpt] SUCCESS: Hoisted DYNAMIC WRITE to Exit Block!");

              Call->eraseFromParent();
              LoopChanged = true;
            }
          }
        }
      }
      return LoopChanged;
    }

    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
      NumFunctionsAnalyzed++;
      bool Changed = false;
      
      AAResults &AA = FAM.getResult<AAManager>(F);
      const DataLayout &DL = F.getParent()->getDataLayout();
      LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
      ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
      DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
      PostDominatorTree &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);
      MemorySSA &MSSA = FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
      
      auto PreorderLoops = LI.getLoopsInPreorder();
      for (Loop *L : PreorderLoops) {
        if (optimiseLoopIO(L, SE, DL, LI, DT, AA, MSSA)) Changed = true;
      }
      
      std::unordered_map<Value*, SmallVector<CallInst*, 8>> ActiveBatches;
      std::unordered_map<Value*, uint64_t> ActiveBatchBytes;

      auto flushAllBatches = [&]() {
	for (auto &Pair : ActiveBatches) {
	  if (flushBatch(Pair.second, F.getParent(), SE, &DT)) Changed = true;
	}
	ActiveBatches.clear();
	ActiveBatchBytes.clear();
      };

      for (BasicBlock &BB : F) {
        for (Instruction &I : llvm::make_early_inc_range(BB)) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
              
            Function *CalleeF = Call->getCalledFunction(); 
            if (CalleeF) {
              StringRef FuncName = CalleeF->getName();
              
              if (FuncName == "fsync" || FuncName == "fdatasync" || FuncName == "sync_file_range" || FuncName == "posix_fadvise" || FuncName == "posix_fadvise64") {
                if (Call->arg_size() > 0) {
		  Value *SyncTarget = Call->getArgOperand(0);
		  Value *BaseFD = getBaseFD(SyncTarget);
		  if (BaseFD && ActiveBatches.count(BaseFD) && !ActiveBatches[BaseFD].empty()) {
		    if (flushBatch(ActiveBatches[BaseFD], F.getParent(), SE, &DT)) Changed = true;
		    ActiveBatchBytes[BaseFD] = 0;
		  }
                }
                continue; 
              } else if (FuncName == "madvise") {
		flushAllBatches();
		continue;
              }
            }

            IOArgs CArgs = getIOArguments(Call, CalleeF);
            bool isWrite = (CArgs.Type == IOArgs::POSIX_WRITE || CArgs.Type == IOArgs::C_FWRITE || CArgs.Type == IOArgs::CXX_WRITE || CArgs.Type == IOArgs::POSIX_PWRITE || CArgs.Type == IOArgs::MPI_WRITE_AT || CArgs.Type == IOArgs::SPLICE || CArgs.Type == IOArgs::SENDFILE || CArgs.Type == IOArgs::IO_SUBMIT || CArgs.Type == IOArgs::AIO_WRITE);
            bool isRead = (CArgs.Type == IOArgs::POSIX_READ || CArgs.Type == IOArgs::C_FREAD || CArgs.Type == IOArgs::POSIX_PREAD || CArgs.Type == IOArgs::MPI_READ_AT || CArgs.Type == IOArgs::CXX_READ);

            if (isWrite || isRead) {
                
              uint64_t CallBytes = 4096; 
              if (CArgs.Length && isa<ConstantInt>(CArgs.Length)) {
		CallBytes = cast<ConstantInt>(CArgs.Length)->getZExtValue();
              } else if (CArgs.Length && SE.isSCEVable(CArgs.Length->getType())) {
		const SCEV *LenSCEV = SE.getSCEV(CArgs.Length);
		auto Max = SE.getUnsignedRangeMax(LenSCEV);
		if (Max.getBitWidth() <= 64 && Max.getZExtValue() < Config.HighWaterMark) {
		  CallBytes = Max.getZExtValue();
		}
              }

              Value *BaseFD = getBaseFD(CArgs.Target);
	      if(BaseFD) {
		auto &Batch = ActiveBatches[BaseFD];
		
		if (!Batch.empty()) {
		  IOArgs BatchArgs = getIOArguments(Batch.front());
		  bool BatchIsRead = (BatchArgs.Type == IOArgs::POSIX_READ || BatchArgs.Type == IOArgs::C_FREAD || BatchArgs.Type == IOArgs::POSIX_PREAD || BatchArgs.Type == IOArgs::MPI_READ_AT || BatchArgs.Type == IOArgs::CXX_READ);
		  if (BatchIsRead != isRead) {
		    if (flushBatch(Batch, F.getParent(), SE, &DT)) Changed = true;
		    ActiveBatchBytes[BaseFD] = 0;
		  }
		}
		if (isSafeToAddToBatch(Batch, Call, AA, DL, SE, DT, PDT)) {

		  if (Batch.size() >= Config.MaxIov) {
		    if (flushBatch(Batch, F.getParent(), SE, &DT)) Changed = true;
		    ActiveBatchBytes[BaseFD] = 0;
		  }

		  Batch.push_back(Call);
		  ActiveBatchBytes[BaseFD] += CallBytes;

		  if (ActiveBatchBytes[BaseFD] >= Config.HighWaterMark) {
		    if (flushBatch(Batch, F.getParent(), SE, &DT)) Changed = true;
		    ActiveBatchBytes[BaseFD] = 0;
		  }
		} else {
		  if (flushBatch(Batch, F.getParent(), SE, &DT)) Changed = true;
		  ActiveBatchBytes[BaseFD] = 0;

		  // If the batch was huge, we already flushed; start fresh with this call
		  Batch.push_back(Call);
		  ActiveBatchBytes[BaseFD] = CallBytes;
		}
	      }
	    }
          }
        }
      }
      
      flushAllBatches();
      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "IOOpt", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
        
      // Command-line parsing for Function-level 'opt -passes=io-opt'
      PB.registerPipelineParsingCallback(
					 [](StringRef Name, FunctionPassManager &FPM, ArrayRef<PassBuilder::PipelineElement>) {
					   if (Name == "io-opt") {
					     FPM.addPass(IOOptimisationPass()); 
					     return true;
					   }
					   return false;
					 });

      // Command-line parsing for our Python Harness 'opt -passes=io-lto-merge'
      PB.registerPipelineParsingCallback(
					 [](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
					   if (Name == "io-lto-merge") {
					     // First, inline cross-file I/O wrappers now that files are merged
					     MPM.addPass(InterProceduralIOBatchingPass());
					     // Then, run the actual batching/vectoring passes
					     FunctionPassManager FPM;
					     FPM.addPass(LoopSimplifyPass());
					     FPM.addPass(LCSSAPass());
					     FPM.addPass(IOOptimisationPass());
					     MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM))); 
					     return true;
					   }
					   return false;
					 });

      // Keep our early pipeline start callback
      PB.registerPipelineStartEPCallback(
					 [](ModulePassManager &MPM, OptimizationLevel Level) {
					   MPM.addPass(InterProceduralIOBatchingPass());
					 });

      // Standard Compile-Time Optimization
      PB.registerOptimizerLastEPCallback(
					 [](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase Phase) {
					   FunctionPassManager FPM;
					   FPM.addPass(LoopSimplifyPass());
					   FPM.addPass(LCSSAPass());
					   FPM.addPass(IOOptimisationPass());
					   MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM))); 
					 });

      // Standard Clang LTO Callback (-flto)
      PB.registerFullLinkTimeOptimizationLastEPCallback(
							[](ModulePassManager &MPM, OptimizationLevel Level) {
							  // Added Interprocedural pass here to catch cross-file I/O wrappers!
							  MPM.addPass(InterProceduralIOBatchingPass());
							  FunctionPassManager FPM;
							  FPM.addPass(LoopSimplifyPass());
							  FPM.addPass(LCSSAPass());
							  FPM.addPass(IOOptimisationPass());
							  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM))); 
							});
    }};
}
