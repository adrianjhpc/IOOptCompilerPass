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

using namespace llvm;

// Expose tunable parameters to the command line
static cl::opt<unsigned> IOBatchThreshold(
    "io-batch-threshold", 
    cl::desc("Minimum number of scattered I/O operations to trigger writev"), 
    cl::init(4));

static cl::opt<unsigned> IOShadowBufferSize(
    "io-shadow-buffer-max", 
    cl::desc("Maximum bytes to allocate on the stack for shadow buffering"), 
    cl::init(4096));
static cl::opt<unsigned> IOHighWaterMark(
    "io-high-water-mark", 
    cl::desc("Maximum cumulative bytes before forcing a pipeline flush (default 64KB)"), 
    cl::init(65536), // 64 KB
    cl::Hidden);

namespace {

  struct IOArgs {
    Value *Target; 
    Value *Buffer; 
    Value *Length; 
    enum { NONE, C_FWRITE, C_FREAD, POSIX_WRITE, POSIX_READ, CXX_WRITE, CXX_READ } Type;
  };

  IOArgs getIOArguments(CallInst *Call) {
    Function *F = Call->getCalledFunction();
    if (!F || !F->hasName() || !F->isDeclaration()) return {nullptr, nullptr, nullptr, IOArgs::NONE};

    std::string Demangled = llvm::demangle(F->getName().str());
    
    if (Demangled == "fwrite") return {Call->getArgOperand(3), Call->getArgOperand(0), Call->getArgOperand(2), IOArgs::C_FWRITE};
    if (Demangled == "fread")  return {Call->getArgOperand(3), Call->getArgOperand(0), Call->getArgOperand(2), IOArgs::C_FREAD};
    if (Demangled == "write")  return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_WRITE};
    if (Demangled == "read")   return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_READ};

    if ((Demangled.find("std::basic_ostream") != std::string::npos || 
         Demangled.find("std::ostream") != std::string::npos) && Demangled.find("::write") != std::string::npos) {
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::CXX_WRITE};
    }
    
    return {nullptr, nullptr, nullptr, IOArgs::NONE};
  }

  bool checkAdjacency(Value *Buf1, Value *Size1, Value *Buf2, const DataLayout &DL) {
    if (auto *GEP = dyn_cast<GEPOperator>(Buf2)) {
      if (GEP->getPointerOperand() == Buf1) {
        APInt Offset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
        if (GEP->accumulateConstantOffset(DL, Offset)) {
          if (auto *ConstSize1 = dyn_cast<ConstantInt>(Size1)) {
            if (Offset == ConstSize1->getValue()) return true; 
          }
        }
      }
    }
    return false;
  }

 bool isSafeToAddToBatch(const std::vector<CallInst*> &Batch, CallInst *NewCall, AAResults &AA, const DataLayout &DL) {
    if (Batch.empty()) return true;
    
    CallInst *LastCall = Batch.back();
    if (LastCall->getCalledFunction() != NewCall->getCalledFunction()) return false;
    
    IOArgs FirstArgs = getIOArguments(Batch.front());
    IOArgs NewArgs = getIOArguments(NewCall);
    
    bool TargetsMatch = (FirstArgs.Target == NewArgs.Target);
    if (!TargetsMatch) {
      auto *Load1 = dyn_cast<LoadInst>(FirstArgs.Target);
      auto *Load2 = dyn_cast<LoadInst>(NewArgs.Target);
      if (Load1 && Load2 && Load1->getPointerOperand() == Load2->getPointerOperand()) {
        TargetsMatch = true;
      }
    }
    if (!TargetsMatch) return false;

    auto getPreciseLoc = [&](Value *Buf, Value *Len) {
      if (auto *C = dyn_cast<ConstantInt>(Len))
        return MemoryLocation(Buf, LocationSize::precise(C->getZExtValue()));
      return MemoryLocation::getBeforeOrAfter(Buf);
    };

    MemoryLocation NewLoc = getPreciseLoc(NewArgs.Buffer, NewArgs.Length);

    for (Instruction *I = LastCall->getNextNode(); I != NewCall; I = I->getNextNode()) {
      if (!I) return false;

      if (auto *CI = dyn_cast<CallInst>(I)) {
          if (CI->getIntrinsicID() == Intrinsic::lifetime_end ||
              CI->getIntrinsicID() == Intrinsic::lifetime_start) {
              return false;
          }
          if (getIOArguments(CI).Type != IOArgs::NONE) return false;

          if (!CI->onlyReadsMemory() && !CI->doesNotAccessMemory()) return false;
      }

      if (!I->mayReadOrWriteMemory()) continue;

      if (isModSet(AA.getModRefInfo(I, NewLoc))) return false;

      for (CallInst *BatchedCall : Batch) {
        IOArgs BArgs = getIOArguments(BatchedCall);
        MemoryLocation BLoc = getPreciseLoc(BArgs.Buffer, BArgs.Length);
        if (isModSet(AA.getModRefInfo(I, BLoc))) return false;
      }
      if (FirstArgs.Target->getType()->isPointerTy()) {
          MemoryLocation TargetLoc(FirstArgs.Target, LocationSize::beforeOrAfterPointer());
          if (isModSet(AA.getModRefInfo(I, TargetLoc))) return false;
      }
    }
    
    return true;
  } 


// ====================================================================
  // 1. Define the I/O Patterns
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
  IOPattern classifyBatch(const std::vector<CallInst*> &Batch, const DataLayout &DL, uint64_t &OutTotalConstSize) {
    if (Batch.size() <= 1) return IOPattern::Unprofitable;

    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isRead = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD);

    // Pattern A: Is it perfectly contiguous?
    bool AllContiguous = true;
    for (size_t i = 0; i < Batch.size() - 1; ++i) {
      IOArgs A = getIOArguments(Batch[i]);
      IOArgs B = getIOArguments(Batch[i+1]);
      if (!checkAdjacency(A.Buffer, A.Length, B.Buffer, DL)) {
        AllContiguous = false;
        break;
      }
    }
    if (AllContiguous) return IOPattern::Contiguous;

    // If it's not contiguous, we can't batch basic reads/writes without vectoring support
    if (FirstArgs.Type != IOArgs::POSIX_WRITE && FirstArgs.Type != IOArgs::POSIX_READ) {
      return IOPattern::Unprofitable;
    }

    // Pattern B: Check for the Strided Pattern (Uniform sizes for SIMD)
    bool isStrided = false;
    uint64_t UniformSize = 0;
    
    if (!isRead && Batch.size() > 1) {
        if (auto *FirstSizeC = dyn_cast<ConstantInt>(getIOArguments(Batch.front()).Length)) {
            UniformSize = FirstSizeC->getZExtValue();
            // Only attempt SIMD gather for standard primitive sizes (e.g., 32-bit or 64-bit fields)
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
        OutTotalConstSize = UniformSize; // Pass the uniform element size to the router
        // Profitability Check: Gathering into a vector is highly profitable if we have >= 4 elements
        if (Batch.size() >= 4 && Batch.size() <= 64) {
            return IOPattern::Strided;
        }
    }

    // Evaluate dynamic sizes and byte-weight
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
    
    // Export the total size so the Router can use it for Shadow Buffering
    OutTotalConstSize = TotalConstSize;

    // Apply the Asymmetric Cost Model
    if (isRead) {
      DynamicThreshold = 2; // Reads always profit from VFS prefetching
    } else {
      Function *F = Batch.back()->getFunction();
      if (F->getInstructionCount() > 150) DynamicThreshold = 3;
      if (AllSizesConstant && TotalConstSize > 128) DynamicThreshold = 3;
    }

    // Pattern C: Vectored I/O (Meets the strict profitable threshold)
    if (Batch.size() >= DynamicThreshold) {
      return IOPattern::Vectored;
    }

    // Pattern D: Shadow Buffer (Fails vectored threshold, but small enough to pack manually)
    if (!isRead && AllSizesConstant && TotalConstSize > 0 && TotalConstSize <= IOShadowBufferSize) {
      return IOPattern::ShadowBuffer;
    }

    // Pattern E: Not profitable
    return IOPattern::Unprofitable;
  }

  // ====================================================================
  // 3. The Router: Executes the IR transformations based on the Classifier
  // ====================================================================
  bool flushBatch(std::vector<CallInst*> &Batch, Module *M) {
    if (Batch.empty()) return false;

    const DataLayout &DL = M->getDataLayout();
    uint64_t TotalConstSize = 0;
    
    // Ask the Classifier what to do!
    IOPattern Pattern = classifyBatch(Batch, DL, TotalConstSize);

    if (Pattern == IOPattern::Unprofitable) {
      Batch.clear();
      return false; // Safely bail out without altering IR
    }

    IRBuilder<> Builder(Batch.back());
    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isRead = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD);

    // Route to the highly specialized emission logic
    switch (Pattern) {
      case IOPattern::Contiguous: {
        Value *TotalLen = FirstArgs.Length;
        for (size_t i = 1; i < Batch.size(); ++i) {
          TotalLen = Builder.CreateAdd(TotalLen, getIOArguments(Batch[i]).Length, "sum.len");
        }
          
        std::vector<Value *> NewArgs;
        if (FirstArgs.Type == IOArgs::C_FWRITE || FirstArgs.Type == IOArgs::C_FREAD) {
          NewArgs = {FirstArgs.Buffer, Batch[0]->getArgOperand(1), TotalLen, FirstArgs.Target};
        } else {
          NewArgs = {FirstArgs.Target, FirstArgs.Buffer, TotalLen};
        }
        Builder.CreateCall(Batch[0]->getCalledFunction(), NewArgs);
        errs() << "[IOOpt] SUCCESS: N-Way merged " << Batch.size() << " contiguous " << (isRead ? "reads" : "writes") << "!\n";
        break;
      }

      case IOPattern::Strided: {
        // TotalConstSize here holds the size of ONE element (e.g., 4 bytes)
        unsigned ElementBytes = TotalConstSize;
        unsigned NumElements = Batch.size();
        
        // Define the LLVM Types for the SIMD Vector
        Type *ElementTy = Builder.getIntNTy(ElementBytes * 8); // e.g., i32
        auto *VecTy = FixedVectorType::get(ElementTy, NumElements);
        
        // Start with an empty (poison) vector register
        Value *GatherVec = PoisonValue::get(VecTy);
        
        // Gather the scattered data into the CPU vector register!
        for (unsigned i = 0; i < NumElements; ++i) {
            IOArgs Args = getIOArguments(Batch[i]);
            Value *DataPtr = Args.Buffer;
            // Load the single struct field
            LoadInst *LoadedVal = Builder.CreateLoad(ElementTy, DataPtr, "strided.load");
            // Insert it into the vector register
            GatherVec = Builder.CreateInsertElement(GatherVec, LoadedVal, Builder.getInt32(i), "gather.insert");
        }
        
        // Allocate a perfectly sized contiguous buffer on the stack
        Function *F = Batch.back()->getFunction();
        IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());
        AllocaInst *ContiguousBuf = EntryBuilder.CreateAlloca(VecTy, nullptr, "simd.shadow.buf");
        
        // Store the entire vector register to the stack in one instruction
        Builder.CreateStore(GatherVec, ContiguousBuf);
        
        // Issue a single write() system call for the entire batch
        Value *TotalLenVal = Builder.getIntN(FirstArgs.Length->getType()->getIntegerBitWidth(), ElementBytes * NumElements);
        Value *BufCast = Builder.CreatePointerCast(ContiguousBuf, Builder.getPtrTy());
        
        std::vector<Value *> NewArgs;
        if (FirstArgs.Type == IOArgs::C_FWRITE) {
            NewArgs = {BufCast, Batch[0]->getArgOperand(1), TotalLenVal, FirstArgs.Target};
        } else {
            NewArgs = {FirstArgs.Target, BufCast, TotalLenVal};
        }
        Builder.CreateCall(Batch[0]->getCalledFunction(), NewArgs);
        
        errs() << "[IOOpt] SUCCESS: SIMD Gathered " << NumElements << " strided writes into 1!\n";
        break;
      }

      case IOPattern::ShadowBuffer: {
        Function *F = Batch.back()->getFunction();
        IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());
        
        Type *Int8Ty = Builder.getInt8Ty();
        ArrayType *ShadowArrTy = ArrayType::get(Int8Ty, TotalConstSize);
        AllocaInst *ShadowBuf = EntryBuilder.CreateAlloca(ShadowArrTy, nullptr, "shadow.buf");
        
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
        } else {
          NewArgs = {FirstArgs.Target, ShadowBuf, TotalLenVal};
        }
        Builder.CreateCall(Batch[0]->getCalledFunction(), NewArgs);
        errs() << "[IOOpt] SUCCESS: Shadow Buffered " << Batch.size() << " writes into 1 (" << TotalConstSize << " bytes)!\n";
        break;
      }

      case IOPattern::Vectored: {
        Type *Int32Ty = Builder.getInt32Ty();
        Type *PtrTy = PointerType::getUnqual(M->getContext());
        Type *SizeTy = DL.getIntPtrType(M->getContext());
        
        StringRef FuncName = isRead ? "readv" : "writev";
        FunctionType *VecTy = FunctionType::get(SizeTy, {Int32Ty, PtrTy, Int32Ty}, false);
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
        Builder.CreateCall(VecFunc, {Fd, IovArray, Builder.getInt32(Batch.size())});
        errs() << "[IOOpt] SUCCESS: N-Way converted " << Batch.size() << " " 
               << (isRead ? "reads" : "writes") << " to " << FuncName << "!\n";
        break;
      }

      case IOPattern::Unprofitable:
      default:
        break;
    }
        
    // Clean up the old unoptimised instructions
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

  bool optimiseLoopIO(Loop *L, ScalarEvolution &SE, const DataLayout &DL) {
    BasicBlock *Preheader = L->getLoopPreheader();
    if (!Preheader) return false;
    unsigned TripCount = SE.getSmallConstantTripCount(L);
    if (TripCount == 0) return false;
    bool Changed = false;
    SCEVExpander Expander(SE, DL, "io.expander");
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : llvm::make_early_inc_range(*BB)) {
        if (auto *Call = dyn_cast<CallInst>(&I)) {
          IOArgs Args = getIOArguments(Call);
          if (Args.Type != IOArgs::POSIX_WRITE && Args.Type != IOArgs::C_FWRITE) continue;
          if (!L->isLoopInvariant(Args.Target)) continue;
          auto *ConstLen = dyn_cast<ConstantInt>(Args.Length);
          if (!ConstLen) continue;
          const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(Args.Buffer));
          if (!AddRec || AddRec->getLoop() != L) continue;
          const SCEVConstant *Step = dyn_cast<SCEVConstant>(AddRec->getStepRecurrence(SE));
          if (Step && Step->getValue()->getValue() == ConstLen->getValue()) {
            Value *BasePtr = Expander.expandCodeFor(AddRec->getStart(), Args.Buffer->getType(), Preheader->getTerminator());
            IRBuilder<> Builder(Preheader->getTerminator());
            Value *NewLen = Builder.getIntN(Args.Length->getType()->getIntegerBitWidth(), TripCount * ConstLen->getZExtValue());
            std::vector<Value *> NewArgs;
            if (Args.Type == IOArgs::C_FWRITE) NewArgs = {BasePtr, Call->getArgOperand(1), NewLen, Args.Target};
            else NewArgs = {Args.Target, BasePtr, NewLen};
            Builder.CreateCall(Call->getCalledFunction(), NewArgs);
            Call->eraseFromParent();
            Changed = true;
          }
        }
      }
    }
    return Changed;
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
    IOOptimisationPass() {}

    // ====================================================================
    // PHASE 1: LOOP-EXIT (LAZY) FLUSHING
    // ====================================================================
    bool optimiseLoopIO(Loop *L, ScalarEvolution &SE, const DataLayout &DL) {
      // 1. We need a guaranteed exit block to drop our batched write into.
      BasicBlock *ExitBB = L->getExitBlock();
      if (!ExitBB) return false;

      // 2. Ask SCEV exactly how many times this loop will run.
      unsigned TripCount = SE.getSmallConstantTripCount(L);
      if (TripCount == 0) return false;

      bool LoopChanged = false;

      // 3. Scan the loop for I/O instructions
      for (BasicBlock *BB : L->blocks()) {
        for (Instruction &I : llvm::make_early_inc_range(*BB)) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            IOArgs Args = getIOArguments(Call);
            
            if (Args.Type == IOArgs::POSIX_WRITE || Args.Type == IOArgs::POSIX_READ || 
                Args.Type == IOArgs::C_FWRITE || Args.Type == IOArgs::C_FREAD) {
                
              auto *ConstLen = dyn_cast<ConstantInt>(Args.Length);
              if (!ConstLen) continue; // Can only scale static payload sizes

              // Calculate the total bytes this loop will write/read across all iterations
              uint64_t ElementSize = ConstLen->getZExtValue();
              uint64_t TotalBytes = ElementSize * TripCount;

              // High-Water Mark protection for massive loops!
              if (TotalBytes > IOHighWaterMark) continue;

              // 4. THE TRANSFORMATION: Lazy Loop-Exit Hoisting!
              IRBuilder<> ExitBuilder(&*ExitBB->getFirstInsertionPt());
              Value *BasePointer = Args.Buffer; 
              Value *TotalLenVal = ExitBuilder.getIntN(Args.Length->getType()->getIntegerBitWidth(), TotalBytes);
              
              std::vector<Value *> NewArgs;
              if (Args.Type == IOArgs::C_FWRITE || Args.Type == IOArgs::C_FREAD) {
                NewArgs = {Args.Buffer, Call->getArgOperand(1), TotalLenVal, Args.Target};
              } else {
                NewArgs = {Args.Target, BasePointer, TotalLenVal};
              }

              ExitBuilder.CreateCall(Call->getCalledFunction(), NewArgs);

              errs() << "[IOOpt] SUCCESS: Hoisted Loop I/O! Scaled " << ElementSize 
                     << " bytes * " << TripCount << " iterations = " << TotalBytes << " bytes at loop exit.\n";

              // 5. Erase the slow I/O call from inside the loop!
              Call->eraseFromParent();
              LoopChanged = true;
            }
          }
        }
      }
      return LoopChanged;
    }

    // ====================================================================
    // MAIN PASS EXECUTION
    // ====================================================================
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
      errs() << "[IOOpt] Analysing function: " << F.getName() << "\n";
      bool Changed = false;

      AAResults &AA = FAM.getResult<AAManager>(F);
      const DataLayout &DL = F.getParent()->getDataLayout();
      LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
      ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);

      // Phase 1: Hoist I/O out of loops
      // We use getLoopsInPreorder to ensure we safely process nested loops
      for (Loop *L : LI.getLoopsInPreorder()) {
        if (optimiseLoopIO(L, SE, DL)) Changed = true;
      }

      // Phase 2: Hoist isolated reads to the top of their blocks
      for (BasicBlock &BB : F) {
        for (Instruction &I : llvm::make_early_inc_range(BB)) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            IOArgs CArgs = getIOArguments(Call);
            if (CArgs.Type == IOArgs::POSIX_READ || CArgs.Type == IOArgs::C_FREAD) {
              if (hoistRead(Call, AA, DL)) Changed = true;
            }
          }
        }
      }

      // Phase 3: Sink isolated writes to the bottom of their blocks
      for (BasicBlock &BB : F) {
        for (Instruction &I : llvm::make_early_inc_range(llvm::reverse(BB))) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            IOArgs CArgs = getIOArguments(Call);
            if (CArgs.Type == IOArgs::POSIX_WRITE || CArgs.Type == IOArgs::C_FWRITE || CArgs.Type == IOArgs::CXX_WRITE) {
              if (sinkWrite(Call, AA, DL)) Changed = true;
            }
          }
        }
      }

      // Phase 4: Block-level batching (Contiguous, Shadow Buffered, Vectored, Strided)
      for (BasicBlock &BB : F) {
        std::vector<CallInst*> IOBatch;
        uint64_t CurrentBatchBytes = 0;

        for (Instruction &I : llvm::make_early_inc_range(BB)) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            IOArgs CArgs = getIOArguments(Call);
            bool isWrite = (CArgs.Type == IOArgs::POSIX_WRITE || CArgs.Type == IOArgs::C_FWRITE || CArgs.Type == IOArgs::CXX_WRITE);
            bool isRead = (CArgs.Type == IOArgs::POSIX_READ || CArgs.Type == IOArgs::C_FREAD);

            if (isWrite || isRead) {
              uint64_t CallBytes = 4096; // Safe default for dynamic sizes
              if (auto *ConstLen = dyn_cast<ConstantInt>(CArgs.Length)) {
                  CallBytes = ConstLen->getZExtValue();
              }

              // Edge Case 1: Return value is used
              if (!Call->use_empty()) {
                if (flushBatch(IOBatch, F.getParent())) Changed = true;
                CurrentBatchBytes = 0;
                continue;
              }

              // Edge Case 2: Switching between Reads and Writes
              if (!IOBatch.empty()) {
                IOArgs BatchArgs = getIOArguments(IOBatch.front());
                bool BatchIsRead = (BatchArgs.Type == IOArgs::POSIX_READ || BatchArgs.Type == IOArgs::C_FREAD);
                if (BatchIsRead != isRead) {
                   if (flushBatch(IOBatch, F.getParent())) Changed = true;
                   CurrentBatchBytes = 0;
                }
              }

              // Main Logic: Is it safe to batch?
              if (isSafeToAddToBatch(IOBatch, Call, AA, DL)) {
                IOBatch.push_back(Call);
                CurrentBatchBytes += CallBytes;

                // High-Water Mark Flush!
                if (CurrentBatchBytes >= IOHighWaterMark) {
                  if (flushBatch(IOBatch, F.getParent())) Changed = true;
                  CurrentBatchBytes = 0;
                }
              } else {
                // Hazard detected.
                if (flushBatch(IOBatch, F.getParent())) Changed = true;
                IOBatch.push_back(Call);
                CurrentBatchBytes = CallBytes; 
              }
            }
          }
        }
        if (flushBatch(IOBatch, F.getParent())) Changed = true;
      }

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
                            if (Name == "write" || Name == "writev" || Name == "fwrite") {
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
