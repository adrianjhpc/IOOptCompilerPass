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

bool flushBatch(std::vector<CallInst*> &Batch, Module *M) {
    if (Batch.size() <= 1) {
      Batch.clear();
      return false;
    }

    const DataLayout &DL = M->getDataLayout();
    bool AllContiguous = true;
    
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
    
    if (!AllContiguous && FirstArgs.Type != IOArgs::POSIX_WRITE && FirstArgs.Type != IOArgs::POSIX_READ) {
        Batch.clear();
        return false;
    }

    bool CanShadowBuffer = false;
    uint64_t TotalConstSize = 0;
    if (!AllContiguous && Batch.size() == 2 && !isRead) {
        auto *C1 = dyn_cast<ConstantInt>(getIOArguments(Batch[0]).Length);
        auto *C2 = dyn_cast<ConstantInt>(getIOArguments(Batch[1]).Length);
        if (C1 && C2) {
            TotalConstSize = C1->getZExtValue() + C2->getZExtValue();
            if (TotalConstSize > 0 && TotalConstSize <= 4096) { 
                CanShadowBuffer = true;
            }
        }
    }

    if (!AllContiguous && Batch.size() < 4 && !CanShadowBuffer) {
        Batch.clear();
        return false;
    }

    IRBuilder<> Builder(Batch.back());

    if (AllContiguous) {
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
        
    } else if (CanShadowBuffer) {
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
      errs() << "[IOOpt] SUCCESS: Shadow Buffered 2 writes into 1 (" << TotalConstSize << " bytes)!\n";

    } else {
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
    
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
      errs() << "[IOOpt] Analyzing function: " << F.getName() << "\n";
      bool Changed = false;
      
      AAResults &AA = FAM.getResult<AAManager>(F);
      const DataLayout &DL = F.getParent()->getDataLayout();
      LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
      ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
      
      for (Loop *L : LI) if (optimiseLoopIO(L, SE, DL)) Changed = true;
    
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
    
      for (BasicBlock &BB : F) {
        std::vector<CallInst*> IOBatch;

        for (Instruction &I : llvm::make_early_inc_range(BB)) {
          if (auto *Call = dyn_cast<CallInst>(&I)) {
            IOArgs CArgs = getIOArguments(Call);
            bool isWrite = (CArgs.Type == IOArgs::POSIX_WRITE || CArgs.Type == IOArgs::C_FWRITE || CArgs.Type == IOArgs::CXX_WRITE);
            bool isRead = (CArgs.Type == IOArgs::POSIX_READ || CArgs.Type == IOArgs::C_FREAD);

            if (isWrite || isRead) {
              if (!Call->use_empty()) {
                if (flushBatch(IOBatch, F.getParent())) Changed = true;
                continue;
              }

              if (!IOBatch.empty()) {
                IOArgs BatchArgs = getIOArguments(IOBatch.front());
                bool BatchIsRead = (BatchArgs.Type == IOArgs::POSIX_READ || BatchArgs.Type == IOArgs::C_FREAD);
                if (BatchIsRead != isRead) {
                   if (flushBatch(IOBatch, F.getParent())) Changed = true;
                }
              }

              if (isSafeToAddToBatch(IOBatch, Call, AA, DL)) {
                IOBatch.push_back(Call);
                if (IOBatch.size() >= 1024) {
                  if (flushBatch(IOBatch, F.getParent())) Changed = true;
                }
              } else {
                if (flushBatch(IOBatch, F.getParent())) Changed = true;
                IOBatch.push_back(Call);
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
