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

  // -----------------------------------------------------------------------------
  // API abstraction and demangling
  // -----------------------------------------------------------------------------
  struct IOArgs {
    Value *Target; 
    Value *Buffer; 
    Value *Length; 
    enum { NONE, C_FWRITE, C_FREAD, POSIX_WRITE, POSIX_READ, CXX_WRITE, CXX_READ } Type;
  };

  IOArgs getIOArguments(CallInst *Call) {
    Function *F = Call->getCalledFunction();
    if (!F || !F->hasName() || !F->isDeclaration()) return {nullptr, nullptr, nullptr, IOArgs::NONE};

    // Demangle the name
    std::string Demangled = llvm::demangle(F->getName().str());
    
    // C Standard Library
    if (Demangled == "fwrite") return {Call->getArgOperand(3), Call->getArgOperand(0), Call->getArgOperand(2), IOArgs::C_FWRITE};
    if (Demangled == "fread")  return {Call->getArgOperand(3), Call->getArgOperand(0), Call->getArgOperand(2), IOArgs::C_FREAD};
    
    // POSIX Calls
    if (Demangled == "write")  return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_WRITE};
    if (Demangled == "read")   return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::POSIX_READ};

    // C++ Standard Library (Catch both the basic_ template and the standard typedefs)
    if ((Demangled.find("std::basic_ostream") != std::string::npos || 
         Demangled.find("std::ostream") != std::string::npos) && 
	Demangled.find("::write") != std::string::npos) {
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::CXX_WRITE};
    }
    
    if ((Demangled.find("std::basic_istream") != std::string::npos || 
         Demangled.find("std::istream") != std::string::npos) && 
	Demangled.find("::read") != std::string::npos) {
      return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2), IOArgs::CXX_READ};
    }

    return {nullptr, nullptr, nullptr, IOArgs::NONE};
  }

  // -----------------------------------------------------------------------------
  // Check memory adjacency
  // -----------------------------------------------------------------------------
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

  // -----------------------------------------------------------------------------
  // Alias analysis and interference 
  // -----------------------------------------------------------------------------
  bool isSafeToAmalgamate(CallInst *Call1, CallInst *Call2, AAResults &AA, const DataLayout &DL) {
    if (Call1->getCalledFunction() != Call2->getCalledFunction()) return false;
    if (!Call1->use_empty() || !Call2->use_empty()) return false;

    IOArgs Args1 = getIOArguments(Call1);
    IOArgs Args2 = getIOArguments(Call2);
    if (!Args1.Target || !Args2.Target || Args1.Target != Args2.Target) return false;

    if (!checkAdjacency(Args1.Buffer, Args1.Length, Args2.Buffer, DL)) return false; 

    LocationSize Size1 = LocationSize::precise(DL.getTypeStoreSize(Args1.Buffer->getType()));
    MemoryLocation Loc1(Args1.Buffer, Size1);
    MemoryLocation Loc2(Args2.Buffer, LocationSize::precise(DL.getTypeStoreSize(Args2.Buffer->getType())));

    for (Instruction *I = Call1->getNextNode(); I != Call2; I = I->getNextNode()) {
      if (!I) return false; 
      if (!I->mayReadOrWriteMemory()) continue;
      if (isModOrRefSet(AA.getModRefInfo(I, Loc1))) return false;
      if (isModOrRefSet(AA.getModRefInfo(I, Loc2))) return false;
    }
    return true; 
  }

  // -----------------------------------------------------------------------------
  // IR builder for amalgamating writes
  // -----------------------------------------------------------------------------
  void mergeWrites(CallInst *Call1, CallInst *Call2) {
    IRBuilder<> Builder(Call2);
    IOArgs Args1 = getIOArguments(Call1);
    IOArgs Args2 = getIOArguments(Call2);

    Value *NewLen = Builder.CreateAdd(Args1.Length, Args2.Length, "merged.io.len");
    std::vector<Value *> NewArgs;
    
    // Reconstruct arguments based on the API abstraction
    if (Args1.Type == IOArgs::C_FWRITE) {
      NewArgs = {Args1.Buffer, Call1->getArgOperand(1), NewLen, Args1.Target};
    } else {
      // POSIX and C++ both use (Target, Buffer, Length)
      NewArgs = {Args1.Target, Args1.Buffer, NewLen};
    }

    Builder.CreateCall(Call1->getCalledFunction(), NewArgs);
    Call1->eraseFromParent();
    Call2->eraseFromParent();
    
    errs() << "Successfully amalgamated writes using demangled C++/POSIX/C APIs!\n";
  }

// -----------------------------------------------------------------------------
  // Hoisting reads (including cross-block CFG traversal)
  // -----------------------------------------------------------------------------
  bool hoistRead(CallInst *ReadCall, AAResults &AA, const DataLayout &DL) {
    IOArgs Args = getIOArguments(ReadCall);
    if (!Args.Buffer) return false;

    LocationSize ReadSize = LocationSize::precise(DL.getTypeStoreSize(Args.Buffer->getType()));
    MemoryLocation DestLoc(Args.Buffer, ReadSize);

    Instruction *InsertPoint = ReadCall;
    Instruction *CurrentInst = ReadCall->getPrevNode();
    BasicBlock *CurrentBB = ReadCall->getParent();
    
    // We replace the simple while loop with an infinite loop that can jump blocks
    while (true) {
      
      // Phase 1: Walk up the current Basic Block
      if (CurrentInst) {
        // We skip Phi nodes and Terminators, but we don't 'break' the whole loop anymore
        if (!CurrentInst->isTerminator() && !isa<PHINode>(CurrentInst)) {
          
          bool DependsOnPrev = false;
          for (Value *Op : ReadCall->operands()) {
            if (Op == CurrentInst) { DependsOnPrev = true; break; }
          }
          if (DependsOnPrev) break; // Hit a strict dependency, must stop hoisting entirely.

          if (CurrentInst->mayReadOrWriteMemory()) {
            if (isModOrRefSet(AA.getModRefInfo(CurrentInst, DestLoc))) break;
            MemoryLocation TargetLoc(Args.Target, LocationSize::beforeOrAfterPointer());
            if (isModSet(AA.getModRefInfo(CurrentInst, TargetLoc))) break;
          }
          
          // Safe to move past this instruction!
          InsertPoint = CurrentInst;
        }
        CurrentInst = CurrentInst->getPrevNode();
      } 
      // Phase 2: We hit the top of the current block. Try to jump to the predecessor
      else {
        // Check if this block have exactly one block that feeds into it.
        BasicBlock *PredBB = CurrentBB->getSinglePredecessor();
        if (!PredBB) break; // Multiple paths lead here (e.g., end of an 'if'). Too dangerous to hoist.

        // Check if the predecessor only feed into our current block
        // (If it branches to other blocks, hoisting would be speculative execution!)
        if (PredBB->getTerminator()->getNumSuccessors() > 1) break;

        // It is a safe linear chain, so we can jump up to the predecessor block.
        CurrentBB = PredBB;
        CurrentInst = CurrentBB->getTerminator(); // Start at the bottom of the new block
      }
    }

    if (InsertPoint != ReadCall) {
      ReadCall->moveBefore(InsertPoint->getIterator());
      errs() << "Successfully hoisted a read call across the CFG\n";
      return true;
    }
    return false;
  }
  
  // ------------------------------------------------------------------------------------------
  // Loop I/O amalgamation for non-unit stride accesses (but still predictable ones) using SCEV
  // ------------------------------------------------------------------------------------------
  bool optimiseLoopIO(Loop *L, ScalarEvolution &SE, const DataLayout &DL) {
    BasicBlock *Preheader = L->getLoopPreheader();
    if (!Preheader) return false;

    unsigned TripCount = SE.getSmallConstantTripCount(L);
    if (TripCount == 0) return false;

    bool Changed = false;

    // Set up the SCEV Expander (used to mathematically reconstruct pointers)
    SCEVExpander Expander(SE, DL, "io.expander");

    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : llvm::make_early_inc_range(*BB)) {
        if (auto *Call = dyn_cast<CallInst>(&I)) {
                
          IOArgs Args = getIOArguments(Call);
          if (Args.Type != IOArgs::C_FWRITE && Args.Type != IOArgs::POSIX_WRITE && Args.Type != IOArgs::CXX_WRITE) continue;

          // Ensure the Target (fd or FILE*) does not change across iterations
          if (!L->isLoopInvariant(Args.Target)) continue;

          auto *ConstLen = dyn_cast<ConstantInt>(Args.Length);
          if (!ConstLen) continue;

          const SCEV *PtrSCEV = SE.getSCEV(Args.Buffer);
          const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(PtrSCEV);
          if (!AddRec || AddRec->getLoop() != L) continue;

          const SCEV *Step = AddRec->getStepRecurrence(SE);

          if (auto *StepConst = dyn_cast<SCEVConstant>(Step)) {
            if (StepConst->getValue()->getValue() == ConstLen->getValue()) {
                        
              uint64_t TotalLenBytes = TripCount * ConstLen->getZExtValue();
                        
              // Use SCEVExpander to retrieve the starting pointer
              // no matter how complex the user's pointer arithmetic was.
	      // This should let us distinguish between separate file pointers.
              Instruction *InsertPt = Preheader->getTerminator();
              Value *BasePtr = Expander.expandCodeFor(AddRec->getStart(), Args.Buffer->getType(), InsertPt);

              IRBuilder<> Builder(InsertPt);
              
              // Match the exact bit-width of the original length parameter
              Value *NewLen = Builder.getIntN(Args.Length->getType()->getIntegerBitWidth(), TotalLenBytes);
                        
              std::vector<Value *> NewArgs;
              if (Args.Type == IOArgs::C_FWRITE) {
                NewArgs = {BasePtr, Call->getArgOperand(1), NewLen, Args.Target};
              } else {
                NewArgs = {Args.Target, BasePtr, NewLen};
              }

              Builder.CreateCall(Call->getCalledFunction(), NewArgs);
              Call->eraseFromParent();
              Changed = true;
            }
          }
        }
      }
    }
    return Changed;
  }

  // -----------------------------------------------------------------------------
  // Main pass
  // -----------------------------------------------------------------------------
  struct IOOptimisationPass : public PassInfoMixin<IOOptimisationPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
      bool Changed = false;
      AAResults &AA = FAM.getResult<AAManager>(F);
      const DataLayout &DL = F.getParent()->getDataLayout();
        
      LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
      ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);

      for (Loop *L : LI) {
	if (optimiseLoopIO(L, SE, DL)) Changed = true;
      }

      for (BasicBlock &BB : F) {
	CallInst *LastWrite = nullptr;

	for (Instruction &I : llvm::make_early_inc_range(BB)) {
	  if (auto *Call = dyn_cast<CallInst>(&I)) {
                    
	    IOArgs Args = getIOArguments(Call);

	    if (Args.Type == IOArgs::C_FWRITE || Args.Type == IOArgs::POSIX_WRITE || Args.Type == IOArgs::CXX_WRITE) {
	      if (LastWrite && isSafeToAmalgamate(LastWrite, Call, AA, DL)) {
		mergeWrites(LastWrite, Call);
		Changed = true;
		LastWrite = nullptr; 
	      } else LastWrite = Call;
	    } 
	    else if (Args.Type == IOArgs::C_FREAD || Args.Type == IOArgs::POSIX_READ || Args.Type == IOArgs::CXX_READ) {
	      if (hoistRead(Call, AA, DL)) Changed = true;
	      LastWrite = nullptr; 
	    } 
	    else LastWrite = nullptr;
	  }
	}
      }
      return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
  };

} // end anonymous namespace

// -----------------------------------------------------------------------------
// Pass plugin registration
// -----------------------------------------------------------------------------
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "IOOptimisationPass", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
					 [](StringRef Name, FunctionPassManager &FPM, ArrayRef<PassBuilder::PipelineElement>) {
					   if (Name == "io-opt") { FPM.addPass(IOOptimisationPass()); return true; }
					   return false;
					 });
      PB.registerOptimizerLastEPCallback(
					 [](ModulePassManager &MPM, OptimizationLevel Level, ThinOrFullLTOPhase Phase) {
					   if (Level != OptimizationLevel::O0) {
					     FunctionPassManager FPM;
					     FPM.addPass(IOOptimisationPass());
					     MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
					   }
					 });
    }};
}
