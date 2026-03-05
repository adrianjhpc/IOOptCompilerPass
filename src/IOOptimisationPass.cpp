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
#include <vector>

using namespace llvm;

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
    
    // SSA Hardened Target Check
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

    // Check all instructions between the last batched write and this new one
    for (Instruction *I = LastCall->getNextNode(); I != NewCall; I = I->getNextNode()) {
      if (!I) return false;
      if (!I->mayReadOrWriteMemory()) continue;
        
      // Does 'I' interfere with the new write's buffer?
      if (isModOrRefSet(AA.getModRefInfo(I, NewLoc))) return false;
        
      // Does 'I' interfere with any buffer already in the batch?
      for (CallInst *BatchedCall : Batch) {
	IOArgs BArgs = getIOArguments(BatchedCall);
	MemoryLocation BLoc = getPreciseLoc(BArgs.Buffer, BArgs.Length);
	if (isModOrRefSet(AA.getModRefInfo(I, BLoc))) return false;
      }
    }
    return true;
  }

bool flushBatch(std::vector<CallInst*> &Batch, Module *M) {
    if (Batch.size() <= 1) {
      Batch.clear();
      return false;
    }

    const DataLayout &DL = M->getDataLayout();
    bool AllContiguous = true;
    
    // Check if the entire chain is contiguous
    for (size_t i = 0; i < Batch.size() - 1; ++i) {
      IOArgs A = getIOArguments(Batch[i]);
      IOArgs B = getIOArguments(Batch[i+1]);
      if (!checkAdjacency(A.Buffer, A.Length, B.Buffer, DL)) {
        AllContiguous = false;
        break;
      }
    }

    IOArgs FirstArgs = getIOArguments(Batch.front());
    bool isRead = (FirstArgs.Type == IOArgs::POSIX_READ || FirstArgs.Type == IOArgs::C_FREAD);
    
    // writev/readv are raw OS system calls. 
    // We cannot use them on C FILE* or C++ std::ostream pointers!
    if (!AllContiguous && FirstArgs.Type != IOArgs::POSIX_WRITE && FirstArgs.Type != IOArgs::POSIX_READ) {
        Batch.clear();
        return false;
    }
    // --------------------------------------------------

    IRBuilder<> Builder(Batch.back());

    if (AllContiguous) {
      // --- N-way contiguous merge functionality
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
        
    } else {
      // --- N-way writev/readv (scatter/gather non-contiguous) merge functionality
      Type *Int32Ty = Builder.getInt32Ty();
      Type *PtrTy = PointerType::getUnqual(M->getContext());
      Type *SizeTy = DL.getIntPtrType(M->getContext());
      
      StringRef FuncName = isRead ? "readv" : "writev";
      FunctionType *VecTy = FunctionType::get(SizeTy, {Int32Ty, PtrTy, Int32Ty}, false);
      FunctionCallee VecFunc = M->getOrInsertFunction(FuncName, VecTy);
      
      StructType *IovecTy = StructType::get(M->getContext(), {PtrTy, SizeTy});
      ArrayType *IovArrayTy = ArrayType::get(IovecTy, Batch.size());
      
      // Allocated the block before the loop
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

// ----------------------------------------------------------------------
  // HELPER: Extends an existing writev/readv call with a new straggler IO
  // ----------------------------------------------------------------------
  bool mergeIntoVectorIO(CallInst *VecCall, CallInst *SingleCall, IRBuilder<> &Builder, Module *M, bool isRead, IOArgs SingleArgs) {
    // We must know the size of the existing vector
    ConstantInt *OldCnt = dyn_cast<ConstantInt>(VecCall->getArgOperand(2));
    if (!OldCnt) return false; 
    
    int OldSize = OldCnt->getZExtValue();
    int NewSize = OldSize + 1;
    if (NewSize >= 1024) return false; // Respect OS IOV_MAX limits
    
    const DataLayout &DL = M->getDataLayout();
    Type *Int32Ty = Builder.getInt32Ty();
    Type *PtrTy = PointerType::getUnqual(M->getContext());
    Type *SizeTy = DL.getIntPtrType(M->getContext());
    
    // Create the new, larger iovec array layout
    StructType *IovecTy = StructType::get(M->getContext(), {PtrTy, SizeTy});
    ArrayType *NewIovArrayTy = ArrayType::get(IovecTy, NewSize);
    
    Function *F = VecCall->getFunction();
    IRBuilder<> EntryBuilder(&F->getEntryBlock(), F->getEntryBlock().begin());
    AllocaInst *NewIovArray = EntryBuilder.CreateAlloca(NewIovArrayTy, nullptr, "iovec.extended");
    
    // 1. Copy the old iovec array into the new one using LLVM MemCpy
    uint64_t IovecSize = DL.getTypeAllocSize(IovecTy);
    Builder.CreateMemCpy(NewIovArray, Align(8), VecCall->getArgOperand(1), Align(8), Builder.getInt64(OldSize * IovecSize));
    
    // 2. Append the straggler's buffer and length to the end
    Value *NewIovPtr = Builder.CreateInBoundsGEP(NewIovArrayTy, NewIovArray, {Builder.getInt32(0), Builder.getInt32(OldSize)});
    Builder.CreateStore(SingleArgs.Buffer, Builder.CreateStructGEP(IovecTy, NewIovPtr, 0));
    Builder.CreateStore(Builder.CreateIntCast(SingleArgs.Length, SizeTy, false), Builder.CreateStructGEP(IovecTy, NewIovPtr, 1));
    
    // 3. Create the replacement vector call
    StringRef FuncName = isRead ? "readv" : "writev";
    FunctionType *VecTy = FunctionType::get(SizeTy, {Int32Ty, PtrTy, Int32Ty}, false);
    FunctionCallee VecFunc = M->getOrInsertFunction(FuncName, VecTy);
    
    Builder.CreateCall(VecFunc, {VecCall->getArgOperand(0), NewIovArray, Builder.getInt32(NewSize)});
    errs() << "[IOOpt] SUCCESS: Straggler merged! " << FuncName << " extended to " << NewSize << " operations.\n";
    
    // 4. Clean up the old instructions
    VecCall->eraseFromParent();
    SingleCall->eraseFromParent();
    return true;
  }
  
  struct IOOptimisationPass : public PassInfoMixin<IOOptimisationPass> {
    // Record when this is running in the LTO phase
    bool IsLTOPhase;
    
    // Constructor defaults to LTO false for standard lit tests
    IOOptimisationPass(bool IsLTOPhase = false) : IsLTOPhase(IsLTOPhase) {}
    
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
      // If we are in the local compile phase, and the Injector tagged this 
      // function for inlining, DO not optimize it yet, save it for LTO.
      if (!IsLTOPhase && F.hasFnAttribute(Attribute::AlwaysInline)) {
	errs() << "[IOOpt] Deferring '" << F.getName() << "' for global LTO phase...\n";
	return PreservedAnalyses::all();
      }
      
      errs() << "[IOOpt] Analyzing function: " << F.getName() << "\n";
      bool Changed = false;
      AAResults &AA = FAM.getResult<AAManager>(F);
      const DataLayout &DL = F.getParent()->getDataLayout();
      LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
      ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
      
      for (Loop *L : LI) if (optimiseLoopIO(L, SE, DL)) Changed = true;
    
      // ====================================================================
      // PHASE 1: Hoist Reads
      // We do this FIRST so that reads are safely grouped at the top of 
      // the blocks before we attempt to calculate adjacency for batching.
      // ====================================================================
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

      // ====================================================================
      // PHASE 2: N-Way Merge (Batching)
      // ====================================================================
      for (BasicBlock &BB : F) {
        std::vector<CallInst*> IOBatch;

        for (Instruction &I : llvm::make_early_inc_range(BB)) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            IOArgs CArgs = getIOArguments(Call);
            
            bool isWrite = (CArgs.Type == IOArgs::POSIX_WRITE || CArgs.Type == IOArgs::C_FWRITE || CArgs.Type == IOArgs::CXX_WRITE);
            bool isRead = (CArgs.Type == IOArgs::POSIX_READ || CArgs.Type == IOArgs::C_FREAD);

            if (isWrite || isRead) {
              
              // If the return value is used, we cannot safely batch it
              // because readv/writev returns a single aggregate size.
              if (!Call->use_empty()) {
                if (flushBatch(IOBatch, F.getParent())) Changed = true;
                continue; 
              }      
              
              // Prevent mixing reads and writes in the same batch
              if (!IOBatch.empty()) {
                IOArgs BatchArgs = getIOArguments(IOBatch.front());
                bool BatchIsRead = (BatchArgs.Type == IOArgs::POSIX_READ || BatchArgs.Type == IOArgs::C_FREAD);
                if (BatchIsRead != isRead) {
                   if (flushBatch(IOBatch, F.getParent())) Changed = true;
                }
              }

              // Batching Logic
              if (isSafeToAddToBatch(IOBatch, Call, AA, DL)) {
                IOBatch.push_back(Call);
                // Prevent hardware limits (POSIX IOV_MAX)
                if (IOBatch.size() >= 1024) {
                  if (flushBatch(IOBatch, F.getParent())) Changed = true;
                }
              } else {
                // Hazard found so flush the current batch and start a new one
                if (flushBatch(IOBatch, F.getParent())) Changed = true;
                IOBatch.push_back(Call);
              }
            }
          }
        }
        // Flush anything left at the end of the BasicBlock
        if (flushBatch(IOBatch, F.getParent())) Changed = true;
      }
      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };
}

// Heuristic injector pass
// This pass scans the whole program before the inliner runs.
// If it finds a small function that does I/O, it forces the compiler to inline it.
// This is to try and expand lto optimisations to things that aren't inlined by heuristic.
struct IOWrapperInlinePass : public PassInfoMixin<IOWrapperInlinePass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
        bool Changed = false;
        
        for (Function &F : M) {
            // Skip external declarations
            if (F.isDeclaration()) continue;

            unsigned InstCount = 0;
            bool HasIO = false;

            // Analyze the function's contents
            for (BasicBlock &BB : F) {
                for (Instruction &I : BB) {
                    InstCount++;
                    if (auto *Call = dyn_cast<CallInst>(&I)) {
                        if (Function *Callee = Call->getCalledFunction()) {
                            StringRef Name = Callee->getName();
                            // Does it call a write function?
                            if (Name == "write" || Name == "writev" || Name == "fwrite") {
                                HasIO = true;
                            }
                        }
                    }
                }
            }

            // If the function has fewer than 30 instructions 
            // it contains an I/O call, it is a wrapper. Force it to inline!
            if (HasIO && InstCount < 30 && !F.hasFnAttribute(Attribute::NoInline)) {
                errs() << "[IOOpt-Injector] Tagging '" << F.getName() << "' for aggressive LTO inlining...\n";
                F.addFnAttr(Attribute::AlwaysInline);
                Changed = true;
            }
        }
        
        return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
};

// -----------------------------------------------------------------------------
// Pass plugin registration
// -----------------------------------------------------------------------------
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "IOOpt", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      
      // 1. lit testing (Local Phase)
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "io-opt") {
            FPM.addPass(IOOptimisationPass(false)); // Local
            return true;
          }
          return false;
        });
      
      // 2. The Injector
      PB.registerPipelineStartEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Level) {
          MPM.addPass(IOWrapperInlinePass());
        });
      
      // 3. Standard Optimizer (Local Phase)
      PB.registerOptimizerLastEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase Phase) {
          MPM.addPass(createModuleToFunctionPassAdaptor(IOOptimisationPass(false))); // Local
        });
      
      // 4. LTO Optimizer (Global Phase!)
      PB.registerFullLinkTimeOptimizationLastEPCallback(
        [](ModulePassManager &MPM, OptimizationLevel Level) {
          MPM.addPass(createModuleToFunctionPassAdaptor(IOOptimisationPass(true))); // LTO!
        });
    }};
}
