#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/PostDominators.h"
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
  bool EnableLogging;

  IOConfig() {
      BatchThreshold = getEnvOrDefault("IO_BATCH_THRESHOLD", 4);
      ShadowBufferSize = getEnvOrDefault("IO_SHADOW_BUFFER_MAX", 4096);
      HighWaterMark = getEnvOrDefault("IO_HIGH_WATER_MARK", 65536);
      EnableLogging = getEnvOrDefault("IO_ENABLE_LOGGING", 1) != 0;
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
    
    if (Demangled == "fwrite") {
        Value *Bytes = getCStreamBytes(Call);
        return Bytes ? IOArgs{Call->getArgOperand(3), Call->getArgOperand(0), Bytes, IOArgs::C_FWRITE} : IOArgs{nullptr, nullptr, nullptr, IOArgs::NONE};
    }
    if (Demangled == "fread") {
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
                              if (PassedFD == LastIOFD || LastIOFD != nullptr) {
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
              InlineFunctionInfo IFI;
              if (InlineFunction(*TargetToInline, IFI).isSuccess()) {
                  LocalChanged = true;
                  Changed = true;
                  NumIPAInlines++;
                  logMessage("[IOOpt] SUCCESS: Inter-Procedural I/O chain merged by inlining wrapper function.");
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

  bool isSafeToAddToBatch(const SmallVectorImpl<CallInst*> &Batch, CallInst *NewCall, AAResults &AA, const DataLayout &DL, ScalarEvolution &SE, DominatorTree &DT, PostDominatorTree &PDT, bool &ForceShadowBuffer) {
    if (Batch.empty()) return true;

    CallInst *LastCall = Batch.back();
    Function *LastCallee = LastCall->getCalledFunction();
    Function *NewCallee = NewCall->getCalledFunction();

    IOArgs FirstArgs = getIOArguments(Batch.front());
    IOArgs LastArgs = getIOArguments(LastCall, LastCallee);
    IOArgs NewArgs = getIOArguments(NewCall, NewCallee);

    // Prevents Null Pointer Dereference on Async Buffers
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

    if (NewArgs.Buffer->getType()->isPointerTy()) {
        MemoryLocation NewLoc = getPreciseLoc(NewArgs.Buffer, NewArgs.Length);
        for (CallInst *BC : Batch) {
            IOArgs BArgs = getIOArguments(BC);
            if (!BArgs.Buffer || !BArgs.Buffer->getType()->isPointerTy()) continue;
            MemoryLocation BLoc = getPreciseLoc(BArgs.Buffer, BArgs.Length);
            if (isReadBatch && !AA.isNoAlias(NewLoc, BLoc)) return false;
        }
    }

    if (!DT.dominates(LastCall, NewCall)) return false;

    if (isReadBatch) {
        for (Value *Op : NewCall->operands()) {
            if (auto *Inst = dyn_cast<Instruction>(Op)) {
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
      if (isReadBatch) return false;
      if (!PDT.dominates(BB2, BB1)) {
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
          if (getIOArguments(CI, Callee).Type != IOArgs::NONE) return true;
          
          // Strict Abort for Opaque Calls (Prevents HTML/Temporal Output Scrambling)
          if (!CI->onlyReadsMemory() && !CI->doesNotAccessMemory()) {
              logMessage("[IOOpt-Debug] Batch Break: Opaque function call may interleave I/O or mutate state.");
              return true; 
          }
      }

      if (FirstArgs.Type == IOArgs::SPLICE || FirstArgs.Type == IOArgs::SENDFILE) return false;

      if (Inst->mayReadOrWriteMemory() && FirstArgs.Buffer->getType()->isPointerTy()) {
          MemoryLocation NewLoc = getPreciseLoc(NewArgs.Buffer, NewArgs.Length);
          
          if (isReadBatch && Inst->mayReadFromMemory()) {
              for (CallInst *BC : Batch) {
                  IOArgs BArgs = getIOArguments(BC);
                  if (!BArgs.Buffer || !BArgs.Buffer->getType()->isPointerTy()) continue;
                  MemoryLocation BLoc = getPreciseLoc(BArgs.Buffer, BArgs.Length);
                  if (isModOrRefSet(AA.getModRefInfo(Inst, BLoc))) {
                      logMessage("[IOOpt-Debug] Batch Break: Read-After-Write dependency on I/O buffer.");
                      return true; 
                  }
              }
          }

          // If a buffer is mutated, we can safely batch it if we force a Shadow Buffer (eager copy).
          if (isModSet(AA.getModRefInfo(Inst, NewLoc))) {
              ForceShadowBuffer = true;
          }

          for (CallInst *BatchedCall : Batch) {
              IOArgs BArgs = getIOArguments(BatchedCall);
              if (!BArgs.Buffer || !BArgs.Buffer->getType()->isPointerTy()) continue;
              MemoryLocation BLoc = getPreciseLoc(BArgs.Buffer, BArgs.Length);
              if (isModSet(AA.getModRefInfo(Inst, BLoc))) {
                  ForceShadowBuffer = true; 
              }
          }

          // But if the File Descriptor pointer itself is mutated, we must break!
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
                          uint64_t &OutTotalRange, ScalarEvolution *SE, bool ForceShadowBuffer) {
    if (Batch.size() < 2) return IOPattern::Unprofitable;

    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isReadBatch = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD || FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::CXX_READ);
    
    if (FirstArgs.Type == IOArgs::SPLICE || FirstArgs.Type == IOArgs::SENDFILE) return IOPattern::Contiguous; 

    if (!ForceShadowBuffer) {
        bool StrictPhysical = true;
        for (size_t i = 0; i < Batch.size() - 1; ++i) {
          if (!checkAdjacency(getIOArguments(Batch[i]).Buffer, getIOArguments(Batch[i]).Length, 
                              getIOArguments(Batch[i+1]).Buffer, DL, SE, false)) {
            StrictPhysical = false;
            break;
          }
        }
        if (StrictPhysical) return IOPattern::Contiguous;
        
        if (Batch.size() >= 2) {
          if (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::POSIX_WRITE || 
              FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::POSIX_PWRITE) {
            return IOPattern::Vectored;
          }
        }
    }

    if (FirstArgs.Type == IOArgs::POSIX_WRITE || FirstArgs.Type == IOArgs::POSIX_PWRITE || 
        FirstArgs.Type == IOArgs::MPI_WRITE_AT || FirstArgs.Type == IOArgs::C_FWRITE || FirstArgs.Type == IOArgs::CXX_WRITE) { 
        
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
      
      if (Batch.size() >= 2) {
          if (!ForceShadowBuffer) return IOPattern::DynamicShadowBuffer;
      }
    }

    // Safely abort if forced shadow buffer on a read
    return IOPattern::Unprofitable;
  }

  bool flushBatch(SmallVectorImpl<CallInst*> &Batch, Module *M, ScalarEvolution &SE, bool ForceShadowBuffer) {
    if (Batch.empty()) return false;

    const DataLayout &DL = M->getDataLayout();
    uint64_t TotalConstSize = 0;
    
    IOPattern Pattern = classifyBatch(Batch, DL, TotalConstSize, &SE, ForceShadowBuffer);

    if (Pattern == IOPattern::Unprofitable) {
      Batch.clear();
      return false; 
    }

    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isRead = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD || FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::MPI_READ_AT || FirstArgs.Type == IOArgs::CXX_READ);
    bool isExplicit = (FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::POSIX_PWRITE);

    Instruction *InsertPt = isRead ? Batch.front() : Batch.back();
    IRBuilder<> InsertBuilder(InsertPt);

    // Guaranteed safe due to API rejection of nulls before flush
    Value *TotalDynLen = InsertBuilder.getIntN(FirstArgs.Length->getType()->getIntegerBitWidth(), 0);
    for (CallInst *C : Batch) {
        Value *L = getIOArguments(C).Length;
        if (L && L->getType() != TotalDynLen->getType()) L = InsertBuilder.CreateZExtOrTrunc(L, TotalDynLen->getType());
        if (L) TotalDynLen = InsertBuilder.CreateAdd(TotalDynLen, L, "dyn.len.add", true, true); 
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
        LoadInst *LoadedVal = InsertBuilder.CreateLoad(ElementTy, Args.Buffer, "strided.load");
        GatherVec = InsertBuilder.CreateInsertElement(GatherVec, LoadedVal, InsertBuilder.getInt32(i), "gather.insert");
      }
      Function *F = Batch.back()->getFunction();
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());
      AllocaInst *ContiguousBuf = EntryBuilder.CreateAlloca(VecTy, nullptr, "simd.shadow.buf");
      ContiguousBuf->setAlignment(Align(4096));
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
      ShadowBuf->setAlignment(Align(4096)); 

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
      Type *PtrTy = InsertBuilder.getPtrTy();
      
      FunctionCallee MemAlignFunc = M->getOrInsertFunction("posix_memalign", InsertBuilder.getInt32Ty(), PtrTy, SizeTy, SizeTy);
      FunctionCallee FreeFunc = M->getOrInsertFunction("free", InsertBuilder.getVoidTy(), PtrTy);
      
      Function *F = Batch.back()->getFunction();
      IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());
      AllocaInst *HeapBufPtr = EntryBuilder.CreateAlloca(PtrTy, nullptr, "dyn.shadow.ptr");
      
      Value *MallocSize = InsertBuilder.CreateZExtOrTrunc(TotalDynLen, SizeTy);
      InsertBuilder.CreateCall(MemAlignFunc, {HeapBufPtr, InsertBuilder.getIntN(SizeTy->getIntegerBitWidth(), 4096), MallocSize});
      Value *HeapBuf = InsertBuilder.CreateLoad(PtrTy, HeapBufPtr, "dyn.shadow.buf");
      
      Value *CurrentOffset = InsertBuilder.getIntN(SizeTy->getIntegerBitWidth(), 0);
      for (size_t i = 0; i < Batch.size(); ++i) {
          CallInst *C = Batch[i];
          IOArgs Args = getIOArguments(C);
          Value *Len = InsertBuilder.CreateZExtOrTrunc(Args.Length, SizeTy);
          Value *DestPtr = InsertBuilder.CreateInBoundsGEP(Int8Ty, HeapBuf, CurrentOffset, "dyn.dest");
          InsertBuilder.CreateMemCpy(DestPtr, Align(1), Args.Buffer, Align(1), Len);
          CurrentOffset = InsertBuilder.CreateAdd(CurrentOffset, Len, "dyn.offset");
      }
      
      MergedCall = InsertBuilder.CreateCall(Batch[0]->getCalledFunction(), buildArgs(HeapBuf));
      InsertBuilder.CreateCall(FreeFunc, {HeapBuf});
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
    
    bool optimiseLoopIO(Loop *L, ScalarEvolution &SE, const DataLayout &DL, LoopAccessInfoManager &LAIs, LoopInfo &LI, DominatorTree &DT, AAResults &AA) {
      
      BasicBlock *Preheader = L->getLoopPreheader();
      BasicBlock *ExitBB = L->getExitBlock();
      if (!Preheader || !ExitBB) return false;

      if (!L->isLoopSimplifyForm() || !L->isLCSSAForm(DT)) {
          return false;
      }

      const LoopAccessInfo &LAI = LAIs.getInfo(*L);
      bool NeedsRuntimeChecks = false;
      const auto *RtPtrChecking = LAI.getRuntimePointerChecking();

      if (!LAI.canVectorizeMemory()) {
          if (!RtPtrChecking || RtPtrChecking->getChecks().empty()) return false;
          NeedsRuntimeChecks = true;
      }

      if (NeedsRuntimeChecks) {
          logMessage("[IOOpt-Debug] Injecting Runtime Pointer Checks for Loop Versioning!");
          LoopVersioning LVer(LAI, RtPtrChecking->getChecks(), L, &LI, &DT, &SE);
          LVer.versionLoop();
          Preheader = L->getLoopPreheader();
          ExitBB = L->getExitBlock();
          if (!Preheader || !ExitBB) return false;
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
              
              Value *TotalLenVal = Expander.expandCodeFor(TotalBytesSCEV, Args.Length->getType(), InsertionPoint);
              
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
      SmallVector<Instruction*, 16> Lifetimes; 
      
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            if (CI->getIntrinsicID() == Intrinsic::lifetime_end) Lifetimes.push_back(CI);
          }
        }
      }
      
      for (Instruction *I : Lifetimes) I->eraseFromParent();
      if (!Lifetimes.empty()) Changed = true;
            
      AAResults &AA = FAM.getResult<AAManager>(F);
      const DataLayout &DL = F.getParent()->getDataLayout();
      LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
      ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
      DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
      PostDominatorTree &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);
      LoopAccessInfoManager &LAIs = FAM.getResult<LoopAccessAnalysis>(F);

      auto PreorderLoops = LI.getLoopsInPreorder();
      for (Loop *L : PreorderLoops) {
        if (optimiseLoopIO(L, SE, DL, LAIs, LI, DT, AA)) Changed = true;
      }
      
      std::unordered_map<Value*, SmallVector<CallInst*, 8>> ActiveBatches;
      std::unordered_map<Value*, uint64_t> ActiveBatchBytes;
      std::unordered_map<Value*, bool> ForceShadowBufMap;

      auto flushAllBatches = [&]() {
          for (auto &Pair : ActiveBatches) {
              if (flushBatch(Pair.second, F.getParent(), SE, ForceShadowBufMap[Pair.first])) Changed = true;
          }
          ActiveBatches.clear();
          ActiveBatchBytes.clear();
          ForceShadowBufMap.clear();
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
                    if (ActiveBatches.count(BaseFD) && !ActiveBatches[BaseFD].empty()) {
                        if (flushBatch(ActiveBatches[BaseFD], F.getParent(), SE, ForceShadowBufMap[BaseFD])) Changed = true;
                        ActiveBatchBytes[BaseFD] = 0;
                        ForceShadowBufMap[BaseFD] = false;
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
              auto &Batch = ActiveBatches[BaseFD];

              if (!Batch.empty()) {
                IOArgs BatchArgs = getIOArguments(Batch.front());
                bool BatchIsRead = (BatchArgs.Type == IOArgs::POSIX_READ || BatchArgs.Type == IOArgs::C_FREAD || BatchArgs.Type == IOArgs::POSIX_PREAD || BatchArgs.Type == IOArgs::MPI_READ_AT || BatchArgs.Type == IOArgs::CXX_READ);
                if (BatchIsRead != isRead) {
                  if (flushBatch(Batch, F.getParent(), SE, ForceShadowBufMap[BaseFD])) Changed = true;
                  ActiveBatchBytes[BaseFD] = 0;
                  ForceShadowBufMap[BaseFD] = false; 
                }
              }

              if (isSafeToAddToBatch(Batch, Call, AA, DL, SE, DT, PDT, ForceShadowBufMap[BaseFD])) {
                Batch.push_back(Call);
                ActiveBatchBytes[BaseFD] += CallBytes;

                if (ActiveBatchBytes[BaseFD] >= Config.HighWaterMark) {
                  if (flushBatch(Batch, F.getParent(), SE, ForceShadowBufMap[BaseFD])) Changed = true;
                  ActiveBatchBytes[BaseFD] = 0;
                  ForceShadowBufMap[BaseFD] = false; 
                }
              } else {
                if (flushBatch(Batch, F.getParent(), SE, ForceShadowBufMap[BaseFD])) Changed = true;
                Batch.push_back(Call);
                ActiveBatchBytes[BaseFD] = CallBytes; 
                ForceShadowBufMap[BaseFD] = false; 
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
      PB.registerPipelineParsingCallback(
                     [](StringRef Name, FunctionPassManager &FPM, ArrayRef<PassBuilder::PipelineElement>) {
                       if (Name == "io-opt") {
                         FPM.addPass(IOOptimisationPass()); 
                         return true;
                       }
                       return false;
                     });
      PB.registerPipelineStartEPCallback(
                     [](ModulePassManager &MPM, OptimizationLevel Level) {
                       MPM.addPass(InterProceduralIOBatchingPass());
                     });
      PB.registerOptimizerLastEPCallback(
                     [](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase Phase) {
                       FunctionPassManager FPM;
                       FPM.addPass(LoopSimplifyPass());
                       FPM.addPass(LCSSAPass());
                       FPM.addPass(IOOptimisationPass());
                       MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM))); 
                     });
      PB.registerFullLinkTimeOptimizationLastEPCallback(
                            [](ModulePassManager &MPM, OptimizationLevel Level) {
                               FunctionPassManager FPM;
                               FPM.addPass(LoopSimplifyPass());
                               FPM.addPass(LCSSAPass());
                               FPM.addPass(IOOptimisationPass());
                               MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM))); 
                            });
    }};
}
