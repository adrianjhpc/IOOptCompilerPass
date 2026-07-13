#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Attributes.h"
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

// NOTE (ABI hack): Clang/LLD do not export these symbols dynamically like 'opt'
// does, so we define them to satisfy the dynamic loader during LTO.
//
// IMPORTANT: These definitions are only safe if this plugin is compiled against
// an LLVM with the *identical* value of LLVM_ENABLE_ABI_BREAKING_CHECKS as the
// host tool (clang/lld) that loads it. A mismatch changes the layout of core
// LLVM data structures and results in silent memory corruption rather than a
// clean load error. Keep the plugin's LLVM build config in lockstep with the host.
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
#include <cerrno>
#include <limits>
#include <mutex>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "io-opt"
STATISTIC(NumLoopsHoisted, "Number of dynamic loop I/Os hoisted safely");
STATISTIC(NumBatchesMerged, "Number of standard I/O batches merged");
STATISTIC(NumZeroCopy, "Number of zero-copy (splice/sendfile) optimizations");
STATISTIC(NumIPAInlines, "Number of inter-procedural I/O chains collapsed");
STATISTIC(NumFunctionsAnalyzed, "Number of functions analyzed by IOOpt");
STATISTIC(NumBatchesRejectedUnsafeUse,
          "Number of batches skipped because a used return was needed pre-merge");

// --- Master enable + gating flags (fix #7: don't transform aggressively just
// because the plugin is loaded). ---
static cl::opt<bool>
    EnableIOOpt("enable-io-opt", cl::init(true), cl::Hidden,
                cl::desc("Enable IOOpt I/O batching/hoisting transforms"));

static cl::opt<bool> EnableEarlyIPO(
    "io-opt-early-ipo", cl::init(false), cl::Hidden,
    cl::desc("Enable IOOpt interprocedural I/O wrapper inlining via *implicit* "
             "pipeline injection (off by default; explicit io-lto-merge always "
             "runs it)"));

// Atomicity / message-boundary caveat. Merging N writes into one writev
// changes atomicity on pipes/FIFOs (PIPE_BUF) and message boundaries on
// datagram/seqpacket sockets. We cannot prove "regular file" from IR, so this is
// the honest lever. Default true preserves batching; set false to be strict.
static cl::opt<bool> AssumeRegularFiles(
    "io-opt-assume-regular-files", cl::init(true), cl::Hidden,
    cl::desc("Assume batched fds are regular files. Disable if pipes, FIFOs, or "
             "datagram/seqpacket sockets may be batched (atomicity/message "
             "boundaries would otherwise change)."));

static unsigned getEnvOrDefaultU(const char *Name, unsigned Default) {
  const char *Val = std::getenv(Name);
  if (!Val || !*Val) return Default;

  errno = 0;
  char *End = nullptr;
  unsigned long X = std::strtoul(Val, &End, 10);

  if (errno != 0 || End == Val || *End != '\0') return Default;
  // NOTE: 0 is intentionally rejected for threshold-style variables so we never
  // operate with a zero threshold. Callers that need a real "0" meaning must not
  // route through this helper.
  if (X == 0 || X > std::numeric_limits<unsigned>::max()) return Default;
  return static_cast<unsigned>(X);
}

struct IOConfig {
  unsigned BatchThreshold;
  unsigned ShadowBufferSize;
  unsigned HighWaterMark;
  unsigned MaxIov;
  bool EnableLogging;

  IOConfig() {
    BatchThreshold   = getEnvOrDefaultU("IO_BATCH_THRESHOLD", 4);
    ShadowBufferSize = getEnvOrDefaultU("IO_SHADOW_BUFFER_MAX", 4096);
    HighWaterMark    = getEnvOrDefaultU("IO_HIGH_WATER_MARK", 65536);
    MaxIov           = getEnvOrDefaultU("IO_MAX_IOV", 1024);
    EnableLogging    = getEnvOrDefaultU("IO_ENABLE_LOGGING", 0) != 0;
  }
};

static IOConfig Config;

static void logMessage(const Twine &Msg) {
  if (Config.EnableLogging) errs() << Msg << "\n";
}

namespace {

  static bool isSymbolName(StringRef Name, StringRef Base) {
    if (Name == Base) return true;
    if (!Name.starts_with(Base)) return false;
    if (Name.size() == Base.size()) return true;
    char Next = Name[Base.size()];
    return Next == '@' || Next == '.';
  }

  // Cache the (expensive) demangle-based C++ stream classification,
  // keyed by the mangled name string (safe across module frees, unlike Function*).
  // 0 = none, 1 = ostream::write, 2 = istream::read.
  static int classifyCxxStreamName(StringRef Name) {
    if (!Name.starts_with("_Z")) return 0;

    static std::mutex CacheMtx;
    static std::unordered_map<std::string, int> Cache;

    std::string Key = Name.str();
    {
      std::lock_guard<std::mutex> Lk(CacheMtx);
      auto It = Cache.find(Key);
      if (It != Cache.end()) return It->second;
    }

    std::string D = llvm::demangle(Key);
    int R = 0;
    if ((D.find("std::basic_ostream") != std::string::npos ||
         D.find("std::ostream") != std::string::npos) &&
        D.find("::write") != std::string::npos) {
      R = 1;
    } else if ((D.find("std::basic_istream") != std::string::npos ||
                D.find("std::istream") != std::string::npos) &&
               D.find("::read") != std::string::npos) {
      R = 2;
    }

    std::lock_guard<std::mutex> Lk(CacheMtx);
    Cache[Key] = R;
    return R;
  }

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
    const IOArgs NONE{nullptr, nullptr, nullptr, IOArgs::NONE};

    auto getCStreamBytes = [](CallInst *CI) -> Value* {
      Value *Size = CI->getArgOperand(1);
      Value *Count = CI->getArgOperand(2);
      if (auto *CSize = dyn_cast<ConstantInt>(Size)) {
        if (CSize->getZExtValue() == 1) return Count;
        if (auto *CCount = dyn_cast<ConstantInt>(Count))
          return ConstantInt::get(Count->getType(),
                                  CSize->getZExtValue() * CCount->getZExtValue());
      }
      if (auto *CCount = dyn_cast<ConstantInt>(Count)) {
        if (CCount->getZExtValue() == 1) return Size;
      }
      return nullptr;
    };

    if (!F) F = Call->getCalledFunction();
    if (!F || !F->hasName() || !F->isDeclaration()) return NONE;

    // Respect nobuiltin so we don't miscompile calls the frontend told
    // us not to treat as the libc function. (User *redefinitions* are already
    // handled because we require F->isDeclaration() above.)
    // NOTE: must use isNoBuiltin(); hasFnAttr(Attribute::NoBuiltin) asserts.
    if (Call->isNoBuiltin()) return NONE;

    StringRef Name = F->getName();

    // Validate argument counts/types before ever indexing operands.
    auto need = [&](unsigned N) { return Call->arg_size() >= N; };
    auto isPtr = [&](unsigned I) {
      return Call->getArgOperand(I)->getType()->isPointerTy();
    };
    auto isInt = [&](unsigned I) {
      return Call->getArgOperand(I)->getType()->isIntegerTy();
    };

    // POSIX pread/pwrite have 4 args (fd, buf, count, offset).
    if (isSymbolName(Name, "pread") || isSymbolName(Name, "pread64")) {
      if (!need(4) || !isPtr(1) || !isInt(2)) return NONE;
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PREAD};
    }
    if (isSymbolName(Name, "pwrite") || isSymbolName(Name, "pwrite64")) {
      if (!need(4) || !isPtr(1) || !isInt(2)) return NONE;
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PWRITE};
    }
    // write/read have 3 args. (Removed bogus "write64"/"read64" matches: those
    // are not real glibc symbols and risked colliding with user functions.)
    if (isSymbolName(Name, "write")) {
      if (!need(3) || !isPtr(1) || !isInt(2)) return NONE;
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_WRITE};
    }
    if (isSymbolName(Name, "read")) {
      if (!need(3) || !isPtr(1) || !isInt(2)) return NONE;
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_READ};
    }

    if (isSymbolName(Name, "fwrite") || isSymbolName(Name, "efwrite")) {
      if (!need(4) || !isPtr(0) || !isInt(1) || !isInt(2) || !isPtr(3)) return NONE;
      Value *Bytes = getCStreamBytes(Call);
      return Bytes ? IOArgs{Call->getArgOperand(3), Call->getArgOperand(0), Bytes, IOArgs::C_FWRITE} : NONE;
    }
    if (isSymbolName(Name, "fread") || isSymbolName(Name, "efread")) {
      if (!need(4) || !isPtr(0) || !isInt(1) || !isInt(2) || !isPtr(3)) return NONE;
      Value *Bytes = getCStreamBytes(Call);
      return Bytes ? IOArgs{Call->getArgOperand(3), Call->getArgOperand(0), Bytes, IOArgs::C_FREAD} : NONE;
    }

    if (isSymbolName(Name, "preadv") || isSymbolName(Name, "preadv2")) {
      if (!need(3) || !isPtr(1)) return NONE;
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PREADV};
    }
    if (isSymbolName(Name, "pwritev") || isSymbolName(Name, "pwritev2")) {
      if (!need(3) || !isPtr(1)) return NONE;
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_PWRITEV};
    }
    if (isSymbolName(Name, "splice")) {
      if (!need(6) || !isInt(4)) return NONE;
      return {Call->getArgOperand(2), Call->getArgOperand(0), Call->getArgOperand(4), IOArgs::SPLICE};
    }
    if (isSymbolName(Name, "sendfile") || isSymbolName(Name, "sendfile64")) {
      if (!need(4) || !isInt(3)) return NONE;
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(3), IOArgs::SENDFILE};
    }
    if (isSymbolName(Name, "io_submit")) {
      if (!need(3)) return NONE;
      return {Call->getArgOperand(0), Call->getArgOperand(2), Call->getArgOperand(1), IOArgs::IO_SUBMIT};
    }
    if (isSymbolName(Name, "aio_write") || isSymbolName(Name, "aio_write64")) {
      if (!need(1) || !isPtr(0)) return NONE;
      return {Call->getArgOperand(0), nullptr, nullptr, IOArgs::AIO_WRITE};
    }

    if (isSymbolName(Name, "MPI_File_write_at") || isSymbolName(Name, "PMPI_File_write_at")) {
      if (!need(6)) return NONE;
      return {Call->getArgOperand(0), Call->getArgOperand(2), Call->getArgOperand(3), IOArgs::MPI_WRITE_AT};
    }
    if (isSymbolName(Name, "MPI_File_read_at") || isSymbolName(Name, "PMPI_File_read_at")) {
      if (!need(6)) return NONE;
      return {Call->getArgOperand(0), Call->getArgOperand(2), Call->getArgOperand(3), IOArgs::MPI_READ_AT};
    }

    // Slow-path: C++ stream wrappers (cached demangle).
    int Cxx = classifyCxxStreamName(Name);
    if (Cxx != 0) {
      if (!need(3)) return NONE;
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2),
              Cxx == 1 ? IOArgs::CXX_WRITE : IOArgs::CXX_READ};
    }

    return NONE;
  }

  struct InterProceduralIOBatchingPass : public PassInfoMixin<InterProceduralIOBatchingPass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
      if (!EnableIOOpt) return PreservedAnalyses::all();

      bool Changed = false;
      bool LocalChanged;

      auto fdKey = [&](Value *V) -> Value * {
        if (!V) return nullptr;
        if (auto *LI = dyn_cast<LoadInst>(V)) {
          Value *Ptr = LI->getPointerOperand();
          if (Ptr->getType()->isPointerTy())
            return const_cast<Value *>(getUnderlyingObject(Ptr));
        }
        return V;
      };

      auto sameFD = [&](Value *A, Value *B) -> bool {
        Value *KA = fdKey(A);
        Value *KB = fdKey(B);
        return KA && KB && KA == KB;
      };

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
                  LastIOFD = fdKey(Args.Target);
                  continue;
                }

                if (IOWrappers.count(Callee)) {
                  int ArgIdx = IOWrappers[Callee];
                  if ((unsigned)ArgIdx >= Call->arg_size()) { LastIOFD = nullptr; continue; }
                  Value *PassedFD = Call->getArgOperand(ArgIdx);
                  if (LastIOFD != nullptr && sameFD(PassedFD, LastIOFD)) {
                    TargetToInline = Call;
                    break;
                  }
                  LastIOFD = fdKey(PassedFD);
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
          Function *Caller = TargetToInline->getFunction();
          Function *Callee = TargetToInline->getCalledFunction();

          std::string CallerName = Caller ? llvm::demangle(Caller->getName().str()) : "unknown";
          std::string CalleeName = Callee ? llvm::demangle(Callee->getName().str()) : "unknown";

          InlineFunctionInfo IFI;
          if (InlineFunction(*TargetToInline, IFI).isSuccess()) {
            LocalChanged = true;
            Changed = true;
            NumIPAInlines++;
            logMessage("[IOOpt-LTO] SUCCESS: Inlined I/O wrapper '" +
                       Twine(CalleeName) + "' into '" + Twine(CallerName) + "'.");
          }
        }
      } while (LocalChanged);

      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };

  // Never zero-extend an integer size SCEV to a *pointer* type.
  // Extend to the pointer's index integer type and use getAddExpr(ptr, int).
  bool checkAdjacency(Value *Buf1, Value *Len1, Value *Buf2, const DataLayout &DL,
                      ScalarEvolution *SE, bool AllowGaps = false) {
    if (SE && Len1 && Buf1->getType()->isPointerTy() && Buf2->getType()->isPointerTy()) {
      const SCEV *Ptr1 = SE->getSCEV(Buf1);
      const SCEV *Ptr2 = SE->getSCEV(Buf2);
      const SCEV *Size1 = SE->getSCEV(Len1);

      if (!isa<SCEVCouldNotCompute>(Ptr1) && !isa<SCEVCouldNotCompute>(Ptr2) &&
          !isa<SCEVCouldNotCompute>(Size1)) {
        Type *IdxTy = DL.getIndexType(Buf1->getType());
        const SCEV *ExtendedSize = SE->getTruncateOrZeroExtend(Size1, IdxTy);
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

  bool isDeeplySafeFromIO(Function *F, SmallPtrSetImpl<Function*> &Visited) {
    if (!F || F->isDeclaration()) return false;
    if (!Visited.insert(F).second) return true;

    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) { if (LI->isVolatile()) return false; }
        else if (auto *SI = dyn_cast<StoreInst>(&I)) { if (SI->isVolatile()) return false; }
        else if (auto *MI = dyn_cast<MemIntrinsic>(&I)) { if (MI->isVolatile()) return false; }

        if (auto *Call = dyn_cast<CallInst>(&I)) {
          Function *SubCallee = Call->getCalledFunction();
          if (getIOArguments(Call, SubCallee).Type != IOArgs::NONE) return false;

          if (SubCallee && SubCallee->hasName()) {
            StringRef N = SubCallee->getName();
            if (N == "fsync" || N == "fdatasync" || N == "msync" || N == "sync_file_range")
              return false;
          }
          if (!isDeeplySafeFromIO(SubCallee, Visited)) return false;
        }
      }
    }
    return true;
  }

  // A batched call's result can be faithfully reconstructed only if
  // every user of that result is dominated by the point where the merged I/O
  // executes (writes: batch's last call; reads: batch's first call). Otherwise
  // the true per-call value isn't available yet at the use site.
  bool insertDominatesAllUses(Instruction *InsertPt, CallInst *C, DominatorTree &DT) {
    for (Use &U : C->uses()) {
      auto *UI = dyn_cast<Instruction>(U.getUser());
      if (!UI) return false;
      if (auto *PN = dyn_cast<PHINode>(UI)) {
        // The value must dominate the end of the corresponding incoming block.
        if (!DT.dominates(InsertPt->getParent(), PN->getIncomingBlock(U)))
          return false;
        continue;
      }
      if (UI->getParent() == InsertPt->getParent()) {
        if (!InsertPt->comesBefore(UI)) return false; // same block, use precedes merge
      } else if (!DT.dominates(InsertPt->getParent(), UI->getParent())) {
        return false;
      }
    }
    return true;
  }

  bool isSafeToAddToBatch(const SmallVectorImpl<CallInst*> &Batch, CallInst *NewCall,
                          AAResults &AA, const DataLayout &DL, ScalarEvolution &SE,
                          DominatorTree &DT, PostDominatorTree &PDT) {
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

      if (LastLen && SE.isSCEVable(LastOffset->getType()) && SE.isSCEVable(NewOffset->getType()) &&
          SE.isSCEVable(LastLen->getType())) {
        const SCEV *SLast = SE.getSCEV(LastOffset);
        const SCEV *SNew = SE.getSCEV(NewOffset);
        const SCEV *SLen = SE.getSCEV(LastLen);
        if (!isa<SCEVCouldNotCompute>(SLast) && !isa<SCEVCouldNotCompute>(SNew) &&
            !isa<SCEVCouldNotCompute>(SLen)) {
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

    // Require the new call's block to post-dominate the last call's block.
    //
    // We previously also admitted *write* batches across a conditional branch
    // whose condition depended on the last call (the "partial-write spoof").
    // That path was dead: for the branch to depend on the last call, the call's
    // return value must be used *before* the merge point (Batch.back()), so the
    // return-availability gate in prepareBatch() -- insertDominatesAllUses() --
    // unconditionally discarded every such batch. Worse, merging there would be
    // semantically wrong: a short/failed early call could no longer suppress the
    // later I/O, because the bytes were already pushed to the fd in one call.
    // Requiring post-dominance makes the accepted set match what we can safely
    // emit.
    if (BB1 != BB2 && !PDT.dominates(BB2, BB1))
      return false;


    LoadInst *Load1 = dyn_cast<LoadInst>(FirstArgs.Target);

    auto checkHazard = [&](Instruction *Inst) -> bool {
      if (auto *CI = dyn_cast<CallInst>(Inst)) {
        Function *Callee = CI->getCalledFunction();
        if (getIOArguments(CI, Callee).Type != IOArgs::NONE) return true;

        if (Callee) {
          if (Callee->isIntrinsic()) {
            Intrinsic::ID ID = Callee->getIntrinsicID();
            if (ID == Intrinsic::dbg_value || ID == Intrinsic::dbg_declare ||
                ID == Intrinsic::dbg_label || ID == Intrinsic::assume) {
              return false;
            }
          }
        }

        if (Callee && Callee->hasName()) {
          StringRef Name = Callee->getName();
          if (Name == "strlen" || Name == "strnlen" || Name == "strcmp" ||
              Name == "htons" || Name == "htonl" || Name == "ntohs" || Name == "ntohl" ||
              Name == "bswap_32" || Name == "bswap_64") {
            return false;
          }
        }

        if (!CI->onlyReadsMemory() && !CI->doesNotAccessMemory()) {
          if (!CI->onlyAccessesArgMemory()) {
            if (Callee && !Callee->isDeclaration()) {
              SmallPtrSet<Function*, 8> Visited;
              if (isDeeplySafeFromIO(Callee, Visited)) {
                return false;
              }
            }
            StringRef BadFuncName = Callee ? Callee->getName() : "indirect_call";
            logMessage("[IOOpt-Debug] Batch Break: Opaque function '" + BadFuncName +
                       "' may interleave I/O or mutate global state.");
            return true;
          }
        }
      }

      if (FirstArgs.Type == IOArgs::SPLICE || FirstArgs.Type == IOArgs::SENDFILE) return false;

      if (Inst->mayReadOrWriteMemory()) {
        if (isReadBatch) {
          if (NewArgs.Buffer->getType()->isPointerTy()) {
            MemoryLocation NewLoc = getPreciseLoc(NewArgs.Buffer, NewArgs.Length);
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
    bool isReadBatch = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD ||
                        FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::CXX_READ);

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

  // Fully-resolved, ready-to-emit batch. All analysis-dependent decisions have
  // already been made, so emission needs no SE/DT.
  struct PreparedBatch {
    SmallVector<CallInst*, 8> Calls;
    IOPattern Pattern = IOPattern::Unprofitable;
    uint64_t TotalRange = 0;
    // Parallel to Calls; only populated when Pattern == Vectored. Each entry is
    // a buffer pointer guaranteed to dominate the batch insert point (expanded
    // here, while SE/DT are valid, if the original did not dominate).
    SmallVector<Value*, 8> VectoredBufs;
  };

  // Resolve a (possibly MaxIov-split) batch into one or more PreparedBatch
  // entries. This is the ONLY place that consumes ScalarEvolution / DominatorTree
  // for batching. It performs SCEV expansion up front so emitPreparedBatch() can
  // run with no analyses.
  //
  // Safety: this only *inserts* IR (via SCEVExpander); it never erases. Therefore
  // SCEVs of pre-existing values remain valid, and dominance among pre-existing
  // instructions is unaffected, so it is safe to prepare every batch before
  // emitting any of them.
  void prepareBatch(SmallVectorImpl<CallInst*> &Batch, Module *M,
                    ScalarEvolution &SE, DominatorTree &DT,
                    std::vector<PreparedBatch> &Out) {
    if (Batch.empty()) return;

    const DataLayout &DL = M->getDataLayout();
    uint64_t TotalRange = 0;
    IOPattern Pattern = classifyBatch(Batch, DL, TotalRange, &SE);
    if (Pattern == IOPattern::Unprofitable) return;

    // Never emit a single vectored call with iovcnt > MaxIov: split first, then
    // re-classify each chunk (a sub-chunk may prefer a different pattern, exactly
    // as the old recursive flush did). Each chunk is validated against its OWN
    // insert point via the recursion below.
    if (Pattern == IOPattern::Vectored) {
      unsigned Limit = std::max(1u, Config.MaxIov);
      if (Batch.size() > Limit) {
        size_t I = 0;
        while (I < Batch.size()) {
          size_t End = I + (size_t)Limit;
          if (End > Batch.size()) End = Batch.size();
          SmallVector<CallInst*, 64> Chunk(Batch.begin() + I, Batch.begin() + End);
          prepareBatch(Chunk, M, SE, DT, Out);
          I = End;
        }
        return;
      }
    }

    // Merge insertion point for THIS (possibly chunked) batch.
    IOArgs FA = getIOArguments(Batch.front());
    bool isRead = (FA.Type == IOArgs::POSIX_READ || FA.Type == IOArgs::C_FREAD ||
                   FA.Type == IOArgs::POSIX_PREAD || FA.Type == IOArgs::MPI_READ_AT ||
                   FA.Type == IOArgs::CXX_READ);
    Instruction *InsertPt = isRead ? Batch.front() : Batch.back();

    // Correctness gate: if any *used* return value would be needed
    // before the merged call executes, we cannot preserve semantics via
    // reconstruction. Leave the calls intact (correct & unoptimized).
    for (CallInst *C : Batch) {
      if (C->use_empty()) continue;
      if (!insertDominatesAllUses(InsertPt, C, DT)) {
        NumBatchesRejectedUnsafeUse++;
        logMessage("[IOOpt] Skipping batch: a used I/O return is consumed before "
                   "the merge point; cannot preserve semantics.");
        return; // do NOT emit
      }
    }

    PreparedBatch PB;
    PB.Calls.assign(Batch.begin(), Batch.end());
    PB.Pattern = Pattern;
    PB.TotalRange = TotalRange;

    if (Pattern == IOPattern::Vectored) {
      SCEVExpander Expander(SE, DL, "io.vectored.expander");
      for (CallInst *C : Batch) {
        IOArgs Args = getIOArguments(C);
        Value *Buf = Args.Buffer;
        if (isa<Instruction>(Buf) && !DT.dominates(cast<Instruction>(Buf), InsertPt)) {
          Buf = Expander.expandCodeFor(SE.getSCEV(Buf), Buf->getType(), InsertPt);
        }
        PB.VectoredBufs.push_back(Buf);
      }
    }

    Out.push_back(std::move(PB));
  }

  // Pure code-gen. Requires NO ScalarEvolution and NO DominatorTree: pattern
  // classification, non-dominating vectored-buffer expansion, and the
  // return-availability gate were all handled in prepareBatch().
  bool emitPreparedBatch(PreparedBatch &PB, Module *M) {
    SmallVectorImpl<CallInst*> &Batch = PB.Calls;
    if (Batch.empty() || PB.Pattern == IOPattern::Unprofitable) return false;

    const DataLayout &DL = M->getDataLayout();
    const uint64_t TotalConstSize = PB.TotalRange;
    const IOPattern Pattern = PB.Pattern;

    Function *ThisF = Batch.back()->getFunction();

    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isRead = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD ||
                   FirstArgs.Type == IOArgs::POSIX_PREAD || FirstArgs.Type == IOArgs::MPI_READ_AT ||
                   FirstArgs.Type == IOArgs::CXX_READ);
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
      IRBuilder<> EntryBuilder(&ThisF->getEntryBlock(), ThisF->getEntryBlock().begin());
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
      IRBuilder<> EntryBuilder(&ThisF->getEntryBlock(), ThisF->getEntryBlock().begin());

      Type *Int8Ty = InsertBuilder.getInt8Ty();
      ArrayType *ShadowArrTy = ArrayType::get(Int8Ty, TotalConstSize);
      AllocaInst *ShadowBuf = EntryBuilder.CreateAlloca(ShadowArrTy, nullptr, "shadow.buf");
      ShadowBuf->setAlignment(Align(64));

      uint64_t CurrentOffset = 0;
      for (size_t i = 0; i < Batch.size(); ++i) {
        CallInst *C = Batch[i];
        IOArgs Args = getIOArguments(C);
        IRBuilder<> CallBuilder(C);
        // Use 64-bit GEP indices to avoid silent truncation.
        Value *DestPtr = CallBuilder.CreateInBoundsGEP(
            ShadowArrTy, ShadowBuf, {CallBuilder.getInt64(0), CallBuilder.getInt64(CurrentOffset)});
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
      PointerType *PtrTy = InsertBuilder.getPtrTy();
      Type *Int32Ty = InsertBuilder.getInt32Ty();
      Type *VoidTy  = InsertBuilder.getVoidTy();

      FunctionType *PosixMemalignTy = FunctionType::get(Int32Ty, {PtrTy, SizeTy, SizeTy}, false);
      FunctionCallee MemAlignFunc = M->getOrInsertFunction("posix_memalign", PosixMemalignTy);

      FunctionType *FreeTy = FunctionType::get(VoidTy, {PtrTy}, false);
      FunctionCallee FreeFunc = M->getOrInsertFunction("free", FreeTy);

      FunctionType *DprintfTy = FunctionType::get(Int32Ty, {Int32Ty, PtrTy}, true);
      FunctionCallee DprintfFn = M->getOrInsertFunction("dprintf", DprintfTy);

      FunctionType *AbortTy = FunctionType::get(VoidTy, {}, false);
      FunctionCallee AbortFn = M->getOrInsertFunction("abort", AbortTy);

      IRBuilder<> EntryBuilder(&ThisF->getEntryBlock(), ThisF->getEntryBlock().begin());

      AllocaInst *HeapBufPtr = EntryBuilder.CreateAlloca(PtrTy, nullptr, "dyn.shadow.ptr");
      HeapBufPtr->setAlignment(Align(alignof(void *)));

      AllocaInst *RCSlot = EntryBuilder.CreateAlloca(Int32Ty, nullptr, "ioopt.pmem.rc");
      RCSlot->setAlignment(Align(4));

      AllocaInst *SizeSlot = EntryBuilder.CreateAlloca(SizeTy, nullptr, "ioopt.pmem.size");
      SizeSlot->setAlignment(Align(alignof(size_t)));

      BasicBlock *OrigBB = InsertPt->getParent();
      BasicBlock *ContBB = OrigBB->splitBasicBlock(InsertPt, "ioopt.dynshadow.cont");
      BasicBlock *TrapBB = BasicBlock::Create(M->getContext(), "ioopt.dynshadow.fail", ThisF, ContBB);

      Instruction *OrigTerm = OrigBB->getTerminator();
      IRBuilder<> PreBuilder(OrigTerm);

      Value *MallocSize = PreBuilder.CreateZExtOrTrunc(TotalDynLen, SizeTy);
      Value *AlignVal = ConstantInt::get(SizeTy, 64);

      Value *RC = PreBuilder.CreateCall(MemAlignFunc, {HeapBufPtr, AlignVal, MallocSize}, "pmem.rc");
      PreBuilder.CreateStore(RC, RCSlot);
      PreBuilder.CreateStore(MallocSize, SizeSlot);

      Value *HeapBuf = PreBuilder.CreateLoad(PtrTy, HeapBufPtr, "dyn.shadow.buf");

      Value *OkRC = PreBuilder.CreateICmpEQ(RC, ConstantInt::get(Int32Ty, 0), "pmem.ok.rc");
      Value *NonNull = PreBuilder.CreateICmpNE(HeapBuf, ConstantPointerNull::get(PtrTy), "pmem.nonnull");
      Value *Ok = PreBuilder.CreateAnd(OkRC, NonNull, "pmem.ok");

      OrigTerm->eraseFromParent();
      BranchInst::Create(ContBB, TrapBB, Ok, OrigBB);

      IRBuilder<> TrapBuilder(TrapBB);
      Value *RCVal = TrapBuilder.CreateLoad(Int32Ty, RCSlot, "ioopt.pmem.rc.val");
      Value *SizeVal = TrapBuilder.CreateLoad(SizeTy, SizeSlot, "ioopt.pmem.size.val");

      Value *Fmt = TrapBuilder.CreateGlobalString(
          "IOOpt: posix_memalign failed (rc=%d, size=%zu)\\n", "ioopt.pmem.fmt");

      TrapBuilder.CreateCall(DprintfFn, {TrapBuilder.getInt32(2), Fmt, RCVal, SizeVal});
      TrapBuilder.CreateCall(AbortFn);
      TrapBuilder.CreateUnreachable();

      IRBuilder<> ContBuilder(&*ContBB->getFirstInsertionPt());

      Value *CurrentOffset = ConstantInt::get(SizeTy, 0);
      for (size_t i = 0; i < Batch.size(); ++i) {
        CallInst *C = Batch[i];
        IOArgs Args = getIOArguments(C);

        Value *Len = ContBuilder.CreateZExtOrTrunc(Args.Length, SizeTy);
        Value *DestPtr = ContBuilder.CreateInBoundsGEP(Int8Ty, HeapBuf, CurrentOffset, "dyn.dest");

        ContBuilder.CreateMemCpy(DestPtr, Align(1), Args.Buffer, Align(1), Len);
        CurrentOffset = ContBuilder.CreateAdd(CurrentOffset, Len, "dyn.offset");
      }

      MergedCall = ContBuilder.CreateCall(Batch[0]->getCalledFunction(), buildArgs(HeapBuf));
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

      IRBuilder<> EntryBuilder(&ThisF->getEntryBlock(), ThisF->getEntryBlock().begin());
      AllocaInst *IovArray = EntryBuilder.CreateAlloca(IovArrayTy, nullptr, "iovec.array.N");
      IovArray->setAlignment(Align(8));

      for (size_t i = 0; i < Batch.size(); ++i) {
        IOArgs Args = getIOArguments(Batch[i]);
        Value *IovPtr = InsertBuilder.CreateInBoundsGEP(IovArrayTy, IovArray, {InsertBuilder.getInt32(0), InsertBuilder.getInt32(i)});

        // Pre-expanded in prepareBatch(); guaranteed to dominate InsertPt.
        Value *SafeBufPtr = PB.VectoredBufs[i];
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

    // A merged call is always inserted before an existing instruction,
    // so it can never be a block terminator. Guard defensively anyway.
    Instruction *AfterMerged = MergedCall->getNextNode();
    assert(AfterMerged && "merged I/O call must not be a block terminator");
    IRBuilder<> RetBuilder(AfterMerged);

    // ---------------------------------------------------------------------
    // Faithful per-call return-value reconstruction.
    //   ret_i = R                    if R < 0 (error)
    //         = clamp(R - P_i, 0, L_i) otherwise
    // fread/fwrite report element COUNT (never negative) -> divide by elem size.
    //
    // IMPORTANT: the RetBuilder anchor (AfterMerged) may itself be one of the
    // batched calls (it IS Batch.front() for read batches). We therefore must
    // NOT erase any batched call while the builder is still live -- doing so
    // frees the anchor and corrupts subsequent insertions. Defer all erasures
    // to a second pass after every instruction has been built.
    // ---------------------------------------------------------------------
    Type *RetTy = MergedCall->getType();
    bool RetIsInt = RetTy->isIntegerTy();
    Value *R = MergedCall;
    Value *IsErr = RetIsInt
        ? RetBuilder.CreateICmpSLT(R, ConstantInt::get(RetTy, 0), "io.iserr")
        : nullptr;
    Value *Prefix = RetIsInt ? ConstantInt::get(RetTy, 0) : nullptr;

    SmallVector<CallInst*, 8> ToErase;
    for (size_t i = 0; i < Batch.size(); ++i) {
      CallInst *C = Batch[i];
      IOArgs CArgs = getIOArguments(C);

      Value *ByteLen = (RetIsInt && CArgs.Length)
          ? RetBuilder.CreateIntCast(CArgs.Length, RetTy, /*isSigned=*/false)
          : nullptr;

      if (C->use_empty()) {
        if (ByteLen) Prefix = RetBuilder.CreateAdd(Prefix, ByteLen);
        ToErase.push_back(C);
        continue;
      }

      Value *Rep = nullptr;

      if (CArgs.Type == IOArgs::CXX_WRITE) {
        Rep = C->getArgOperand(0);                       // ostream& (the stream)
      } else if (CArgs.Type == IOArgs::MPI_WRITE_AT || CArgs.Type == IOArgs::MPI_READ_AT) {
        Rep = RetBuilder.getInt32(0);                    // MPI_SUCCESS
      } else if (RetIsInt && ByteLen) {
        // clamp(R - Prefix, 0, ByteLen)
        Value *Avail  = RetBuilder.CreateSub(R, Prefix, "io.avail");
        Value *NegLo  = RetBuilder.CreateICmpSLT(Avail, ConstantInt::get(RetTy, 0));
        Value *Lo     = RetBuilder.CreateSelect(NegLo, ConstantInt::get(RetTy, 0), Avail);
        Value *OverHi = RetBuilder.CreateICmpSGT(Lo, ByteLen);
        Value *Bytes  = RetBuilder.CreateSelect(OverHi, ByteLen, Lo, "io.bytes");

        if (CArgs.Type == IOArgs::C_FWRITE || CArgs.Type == IOArgs::C_FREAD) {
          // fread/fwrite return element COUNT, and never go negative.
          Value *Elem     = RetBuilder.CreateIntCast(C->getArgOperand(1), RetTy, false);
          Value *ElemZero = RetBuilder.CreateICmpEQ(Elem, ConstantInt::get(RetTy, 0));
          Value *SafeElem = RetBuilder.CreateSelect(ElemZero, ConstantInt::get(RetTy, 1), Elem);
          Value *Items    = RetBuilder.CreateUDiv(Bytes, SafeElem, "io.items");
          Rep = RetBuilder.CreateSelect(ElemZero, ConstantInt::get(RetTy, 0), Items);
        } else {
          // POSIX byte-oriented: propagate the negative error code unchanged.
          Rep = RetBuilder.CreateSelect(IsErr, R, Bytes, "io.posix.ret");
        }

        if (C->getType() != Rep->getType())
          Rep = RetBuilder.CreateIntCast(Rep, C->getType(), /*isSigned=*/true);
      } else {
        Rep = R;
        if (C->getType() != Rep->getType())
          Rep = RetBuilder.CreateIntCast(Rep, C->getType(), true);
      }

      if (ByteLen) Prefix = RetBuilder.CreateAdd(Prefix, ByteLen);
      C->replaceAllUsesWith(Rep);   // RAUW does not move/erase C -> anchor safe
      ToErase.push_back(C);
    }

    // Second pass: now that the builder is no longer used, it is safe to erase.
    for (CallInst *C : ToErase)
      C->eraseFromParent();

    return true;

  }

  // ------------------------------------------------------------------
  // Split loop hoisting and batching into two distinct passes so the
  // pass manager recomputes SE/DT/MSSA/AA between them (the hoist pass returns
  // PreservedAnalyses::none() when it changes anything).
  // ------------------------------------------------------------------

  struct IOLoopHoistingPass : public PassInfoMixin<IOLoopHoistingPass> {
    static Value *getMemoryWritePtr(Instruction &I) {
      if (auto *SI = dyn_cast<StoreInst>(&I)) return SI->getPointerOperand();
      if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) return RMW->getPointerOperand();
      if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) return CX->getPointerOperand();
      if (auto *MT = dyn_cast<MemTransferInst>(&I)) return MT->getDest();
      if (auto *MS = dyn_cast<MemSetInst>(&I)) return MS->getDest();
      return nullptr;
    }

    static bool isSafeToHoistLoopIOCall(CallInst *Call, const IOArgs &Args, Loop *L,
                                        ScalarEvolution &SE, const DataLayout &DL,
                                        AAResults &AA, DominatorTree &DT, MemorySSA &MSSA) {
      if (!Args.Buffer || !Args.Length) return false;
      if (!isa<ConstantInt>(Args.Length)) return false;

      auto *LenC = cast<ConstantInt>(Args.Length);
      uint64_t ElemLen = LenC->getZExtValue();
      if (ElemLen == 0) return false;

      const SCEV *BackedgeCount = SE.getBackedgeTakenCount(L);
      if (isa<SCEVCouldNotCompute>(BackedgeCount)) return false;
      auto *BEC = dyn_cast<SCEVConstant>(BackedgeCount);
      if (!BEC) return false;

      uint64_t Trips = BEC->getAPInt().getZExtValue() + 1;
      if (Trips == 0) return false;
      if (Trips > (std::numeric_limits<uint64_t>::max() / ElemLen)) return false;
      uint64_t TotalLen = Trips * ElemLen;

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

      MemoryLocation FullRange(BasePtr, LocationSize::precise(TotalLen));

      for (BasicBlock *BB : L->blocks()) {
        for (Instruction &I : *BB) {
          if (&I == Call) continue;
          if (!I.mayWriteToMemory()) continue;

          MemoryAccess *MA = MSSA.getMemoryAccess(&I);
          if (!MA) return false;
          if (!isa<MemoryDef>(MA)) continue;

          if (!isModSet(AA.getModRefInfo(&I, FullRange))) continue;
          if (!DT.dominates(&I, Call)) return false;

          Value *WPtr = getMemoryWritePtr(I);
          if (!WPtr) return false;

          const SCEV *WS = SE.getSCEV(WPtr);
          auto *WAR = dyn_cast<SCEVAddRecExpr>(WS);
          if (!WAR || WAR->getLoop() != L) return false;

          const SCEV *WStep = SE.getTruncateOrZeroExtend(WAR->getStepRecurrence(SE),
                                                         DL.getIntPtrType(Call->getContext()));
          if (!SE.isKnownNonNegative(WStep)) return false;
          if (!SE.isKnownPredicate(ICmpInst::ICMP_EQ, WStep, BufStep)) return false;

          const SCEV *StartDiff = SE.getMinusSCEV(WAR->getStart(), BufAR->getStart());
          auto *CD = dyn_cast<SCEVConstant>(StartDiff);
          if (!CD) return false;
          uint64_t Off = CD->getAPInt().getZExtValue();
          if (Off >= ElemLen) return false;
        }
      }

      return true;
    }

    static bool optimiseLoopIO(Loop *L, ScalarEvolution &SE, const DataLayout &DL,
                               LoopInfo &LI, DominatorTree &DT, AAResults &AA, MemorySSA &MSSA) {
      BasicBlock *Preheader = L->getLoopPreheader();
      BasicBlock *ExitBB = L->getExitBlock();
      if (!Preheader || !ExitBB) return false;

      if (!L->isLoopSimplifyForm() || !L->isLCSSAForm(DT)) return false;

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
              if (!isSafeToHoistLoopIOCall(Call, Args, L, SE, DL, AA, DT, MSSA)) continue;

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
                    if (getIOArguments(ScanCall).Type != IOArgs::NONE ||
                        (!ScanCall->onlyReadsMemory() && !ScanCall->doesNotAccessMemory())) {
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
              logMessage(isRead ? "[IOOpt] SUCCESS: Hoisted DYNAMIC READ to Preheader!"
                                : "[IOOpt] SUCCESS: Hoisted DYNAMIC WRITE to Exit Block!");

              Call->eraseFromParent();
              LoopChanged = true;
            }
          }
        }
      }
      return LoopChanged;
    }


    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
      if (!EnableIOOpt) return PreservedAnalyses::all();

      // Collapsing a loop of reads/writes into one large call changes PIPE_BUF atomicity on
      // pipes/FIFOs and message boundaries on datagram/seqpacket sockets. We can't
      // prove "regular file" from IR, so honour the same lever batching uses.
      // (Applies to both reads and writes: a merged datagram read also changes
      // per-message boundary semantics.)
      if (!AssumeRegularFiles) return PreservedAnalyses::all();

      const DataLayout &DL = F.getParent()->getDataLayout();
      bool Changed = false;


      // After any mutating hoist, invalidate + refetch analyses so we
      // never make a subsequent decision based on stale SE/DT/MSSA/LI/AA.
      bool Again = true;
      while (Again) {
        Again = false;

        LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
        ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
        DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
        AAResults &AA = FAM.getResult<AAManager>(F);
        MemorySSA &MSSA = FAM.getResult<MemorySSAAnalysis>(F).getMSSA();

        for (Loop *L : LI.getLoopsInPreorder()) {
          if (optimiseLoopIO(L, SE, DL, LI, DT, AA, MSSA)) {
            Changed = true;
            Again = true;
            FAM.invalidate(F, PreservedAnalyses::none());
            break; // refetch fresh analyses before touching more loops
          }
        }
      }

      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };

  struct IOBatchingPass : public PassInfoMixin<IOBatchingPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
      if (!EnableIOOpt) return PreservedAnalyses::all();

      NumFunctionsAnalyzed++;

      AAResults &AA = FAM.getResult<AAManager>(F);
      const DataLayout &DL = F.getParent()->getDataLayout();
      ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
      DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
      PostDominatorTree &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);

      // Three-phase design.
      //  Phase 1 (decision): walk read-only, gather closed batches. AA/DT/PDT/SE valid.
      //  Phase 2 (prepare):  classify, apply the return-availability gate, and expand
      //                      vectored buffers. Insert-only, so SE/DT facts about
      //                      pre-existing values stay valid across batches.
      //  Phase 3 (emit):     pure code-gen, consults no analyses at all.
      std::vector<SmallVector<CallInst*, 8>> Completed;
      std::unordered_map<Value*, SmallVector<CallInst*, 8>> ActiveBatches;
      std::unordered_map<Value*, uint64_t> ActiveBatchBytes;

      auto closeBatch = [&](Value *FD) {
        auto It = ActiveBatches.find(FD);
        if (It != ActiveBatches.end() && !It->second.empty()) {
          Completed.push_back(std::move(It->second));
          It->second.clear();
        }
        ActiveBatchBytes[FD] = 0;
      };

      auto closeAll = [&]() {
        for (auto &Pair : ActiveBatches) {
          if (!Pair.second.empty()) {
            Completed.push_back(std::move(Pair.second));
            Pair.second.clear();
          }
        }
        ActiveBatchBytes.clear();
      };

      // Phase 1
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          auto *Call = dyn_cast<CallInst>(&I);
          if (!Call) continue;

          Function *CalleeF = Call->getCalledFunction();
          if (CalleeF) {
            StringRef FuncName = CalleeF->getName();

            if (FuncName == "fsync" || FuncName == "fdatasync" || FuncName == "sync_file_range" ||
                FuncName == "posix_fadvise" || FuncName == "posix_fadvise64" || FuncName == "msync" ||
                FuncName == "close" || FuncName == "fclose" || FuncName == "fflush") {
              if (Call->arg_size() > 0) {
                Value *SyncTarget = Call->getArgOperand(0);
                Value *BaseFD = getBaseFD(SyncTarget);
                if (BaseFD) closeBatch(BaseFD);
              }
              continue;
            } else if (FuncName == "madvise") {
              closeAll();
              continue;
            }
          }

          IOArgs CArgs = getIOArguments(Call, CalleeF);
          bool isWrite = (CArgs.Type == IOArgs::POSIX_WRITE || CArgs.Type == IOArgs::C_FWRITE || CArgs.Type == IOArgs::CXX_WRITE || CArgs.Type == IOArgs::POSIX_PWRITE || CArgs.Type == IOArgs::MPI_WRITE_AT || CArgs.Type == IOArgs::SPLICE || CArgs.Type == IOArgs::SENDFILE || CArgs.Type == IOArgs::IO_SUBMIT || CArgs.Type == IOArgs::AIO_WRITE);
          bool isRead = (CArgs.Type == IOArgs::POSIX_READ || CArgs.Type == IOArgs::C_FREAD || CArgs.Type == IOArgs::POSIX_PREAD || CArgs.Type == IOArgs::MPI_READ_AT || CArgs.Type == IOArgs::CXX_READ);

          if (!isWrite && !isRead) continue;

          // If we cannot assume regular files, merging would change
          // atomicity/message boundaries on pipes/sockets. Skip batching.
          if (!AssumeRegularFiles) continue;

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
          if (!BaseFD) continue;

          {
            SmallVector<CallInst*, 8> &Batch = ActiveBatches[BaseFD];
            if (!Batch.empty()) {
              IOArgs BatchArgs = getIOArguments(Batch.front());
              bool BatchIsRead = (BatchArgs.Type == IOArgs::POSIX_READ || BatchArgs.Type == IOArgs::C_FREAD || BatchArgs.Type == IOArgs::POSIX_PREAD || BatchArgs.Type == IOArgs::MPI_READ_AT || BatchArgs.Type == IOArgs::CXX_READ);
              if (BatchIsRead != isRead) closeBatch(BaseFD);
            }
          }

          SmallVector<CallInst*, 8> &Batch = ActiveBatches[BaseFD];
          if (isSafeToAddToBatch(Batch, Call, AA, DL, SE, DT, PDT)) {
            if (Batch.size() >= Config.MaxIov) closeBatch(BaseFD);

            SmallVector<CallInst*, 8> &B = ActiveBatches[BaseFD];
            B.push_back(Call);
            ActiveBatchBytes[BaseFD] += CallBytes;

            if (ActiveBatchBytes[BaseFD] >= Config.HighWaterMark) closeBatch(BaseFD);
          } else {
            closeBatch(BaseFD);
            ActiveBatches[BaseFD].push_back(Call);
            ActiveBatchBytes[BaseFD] = CallBytes;
          }
        }
      }

      closeAll();

      // Phase 2: resolve every batch while SE/DT are still valid.
      std::vector<PreparedBatch> Prepared;
      for (auto &B : Completed)
        prepareBatch(B, F.getParent(), SE, DT, Prepared);

      // Phase 3: pure emission; no analyses consulted.
      bool Changed = false;
      for (auto &PB : Prepared)
        if (emitPreparedBatch(PB, F.getParent())) Changed = true;

      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "IOOpt", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {

      auto addFunctionPipeline = [](ModulePassManager &MPM) {
        FunctionPassManager FPM;
        FPM.addPass(LoopSimplifyPass());
        FPM.addPass(LCSSAPass());
        FPM.addPass(IOLoopHoistingPass());   // separate pass -> analyses recomputed
        FPM.addPass(IOBatchingPass());
        MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
      };

      // opt -passes=io-opt  (function-level)
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM, ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "io-opt") {
            FPM.addPass(IOLoopHoistingPass());
            FPM.addPass(IOBatchingPass());
            return true;
          }
          return false;
        });

      // opt -passes=io-lto-merge  (module-level)
      PB.registerPipelineParsingCallback(
        [addFunctionPipeline](StringRef Name, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "io-lto-merge") {
            // Explicitly requested by the user: always run the interprocedural
            // wrapper-inlining step (its whole reason for existing). EnableEarlyIPO
            // only governs *implicit* injection at pipeline start.
            MPM.addPass(InterProceduralIOBatchingPass());
            addFunctionPipeline(MPM);
            return true;
          }
          return false;
        });

      // Early IPO wrapper inlining is opt-in for *implicit* injection (fix #7).
      PB.registerPipelineStartEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Level) {
          if (EnableEarlyIPO)
            MPM.addPass(InterProceduralIOBatchingPass());
        });

      PB.registerOptimizerLastEPCallback(
        [addFunctionPipeline](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase Phase) {
          addFunctionPipeline(MPM);
        });

      // Standard Clang LTO (-flto): preserve the original behaviour of always
      // running the interprocedural inliner so cross-file wrappers are caught.
      PB.registerFullLinkTimeOptimizationLastEPCallback(
        [addFunctionPipeline](ModulePassManager &MPM, OptimizationLevel Level) {
          MPM.addPass(InterProceduralIOBatchingPass());
          addFunctionPipeline(MPM);
        });
    }};
}

