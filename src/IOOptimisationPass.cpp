#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include <vector>

using namespace llvm;

namespace {

// -----------------------------------------------------------------------------
// Helper 1: API Abstraction for POSIX vs C-Standard I/O
// -----------------------------------------------------------------------------
struct IOArgs {
    Value *Target; // FILE* or fd (int)
    Value *Buffer; // The memory pointer
    Value *Length; // The size of the read/write
};

IOArgs getIOArguments(CallInst *Call) {
    StringRef Name = Call->getCalledFunction()->getName();
    
    if (Name == "fwrite" || Name == "fread") {
        // Signature: ptr buf, size_t size, size_t nmemb, FILE *stream
        // Assuming size is 1, nmemb (Arg 2) is the length
        return {Call->getArgOperand(3), Call->getArgOperand(0), Call->getArgOperand(2)};
    } 
    else if (Name == "write" || Name == "read") {
        // Signature: int fd, ptr buf, size_t count
        return {Call->getArgOperand(0), Call->getArgOperand(1), Call->getArgOperand(2)};
    }
    
    return {nullptr, nullptr, nullptr};
}

// -----------------------------------------------------------------------------
// Helper 2: Check Memory Adjacency using GEPs
// -----------------------------------------------------------------------------
bool checkAdjacency(Value *Buf1, Value *Size1, Value *Buf2, const DataLayout &DL) {
    if (auto *GEP = dyn_cast<GEPOperator>(Buf2)) {
        if (GEP->getPointerOperand() == Buf1) {
            APInt Offset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
            if (GEP->accumulateConstantOffset(DL, Offset)) {
                if (auto *ConstSize1 = dyn_cast<ConstantInt>(Size1)) {
                    if (Offset == ConstSize1->getValue()) {
                        return true; 
                    }
                }
            }
        }
    }
    return false;
}

// -----------------------------------------------------------------------------
// Helper 3: Alias Analysis & Interference Logic for Writes
// -----------------------------------------------------------------------------
bool isSafeToAmalgamate(CallInst *Call1, CallInst *Call2, AAResults &AA, const DataLayout &DL) {
    if (Call1->getCalledFunction() != Call2->getCalledFunction()) return false;
    
    // We cannot easily merge them if the return values are used by the program
    if (!Call1->use_empty() || !Call2->use_empty()) return false;

    IOArgs Args1 = getIOArguments(Call1);
    IOArgs Args2 = getIOArguments(Call2);
    
    if (!Args1.Target || !Args2.Target || Args1.Target != Args2.Target) return false;

    // ADJACENCY CHECK
    if (!checkAdjacency(Args1.Buffer, Args1.Length, Args2.Buffer, DL)) {
        return false; 
    }

    // INTERFERENCE CHECK
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
// Helper 4: IR Builder for Amalgamating Writes
// -----------------------------------------------------------------------------
void mergeWrites(CallInst *Call1, CallInst *Call2) {
    IRBuilder<> Builder(Call2);
    IOArgs Args1 = getIOArguments(Call1);
    IOArgs Args2 = getIOArguments(Call2);

    Value *NewLen = Builder.CreateAdd(Args1.Length, Args2.Length, "merged.io.len");
    
    StringRef Name = Call1->getCalledFunction()->getName();
    std::vector<Value *> NewArgs;
    
    // Reconstruct the argument list based on the target API
    if (Name == "fwrite") {
        NewArgs = {Args1.Buffer, Call1->getArgOperand(1), NewLen, Args1.Target};
    } else if (Name == "write") {
        NewArgs = {Args1.Target, Args1.Buffer, NewLen};
    }

    Builder.CreateCall(Call1->getCalledFunction(), NewArgs);

    Call1->eraseFromParent();
    Call2->eraseFromParent();
    
    errs() << "Successfully amalgamated two " << Name << " calls!\n";
}

// -----------------------------------------------------------------------------
// Helper 5: Hoisting Reads with Dominance and Interference Checks
// -----------------------------------------------------------------------------
bool hoistRead(CallInst *ReadCall, AAResults &AA, const DataLayout &DL) {
    IOArgs Args = getIOArguments(ReadCall);
    if (!Args.Buffer) return false;

    LocationSize ReadSize = LocationSize::precise(DL.getTypeStoreSize(Args.Buffer->getType()));
    MemoryLocation DestLoc(Args.Buffer, ReadSize);

    Instruction *InsertPoint = ReadCall;
    Instruction *Prev = ReadCall->getPrevNode();
    
    while (Prev) {
        if (Prev->isTerminator() || isa<PHINode>(Prev)) break;

        // DOMINANCE CHECK: Does 'Prev' define an operand we need?
        bool DependsOnPrev = false;
        for (Value *Op : ReadCall->operands()) {
            if (Op == Prev) {
                DependsOnPrev = true;
                break;
            }
        }
        if (DependsOnPrev) break; // We cannot hoist past our own definitions!

        // INTERFERENCE CHECK
        if (Prev->mayReadOrWriteMemory()) {
            if (isModOrRefSet(AA.getModRefInfo(Prev, DestLoc))) break;
            
            MemoryLocation TargetLoc(Args.Target, LocationSize::beforeOrAfterPointer());
            if (isModSet(AA.getModRefInfo(Prev, TargetLoc))) break;
        }

        InsertPoint = Prev;
        Prev = Prev->getPrevNode();
    }

    if (InsertPoint != ReadCall) {
        // Use getIterator() to safely preserve debug metadata
        ReadCall->moveBefore(InsertPoint->getIterator());
        errs() << "Successfully hoisted a " << ReadCall->getCalledFunction()->getName() << " call!\n";
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
// The Main Pass
// -----------------------------------------------------------------------------
struct IOOptimizationPass : public PassInfoMixin<IOOptimizationPass> {
    
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
        bool Changed = false;
        AAResults &AA = FAM.getResult<AAManager>(F);
        const DataLayout &DL = F.getParent()->getDataLayout();

        for (BasicBlock &BB : F) {
            CallInst *LastWrite = nullptr;

            for (Instruction &I : llvm::make_early_inc_range(BB)) {
                if (auto *Call = dyn_cast<CallInst>(&I)) {
                    Function *CalledFn = Call->getCalledFunction();
                    
                    // GUARD: Must have a name and be a system/external declaration
                    if (!CalledFn || !CalledFn->hasName() || !CalledFn->isDeclaration()) {
                        LastWrite = nullptr; 
                        continue;
                    }

                    StringRef FnName = CalledFn->getName();

                    // Handle Write Amalgamation
                    if (FnName == "fwrite" || FnName == "write") {
                        if (LastWrite && isSafeToAmalgamate(LastWrite, Call, AA, DL)) {
                            mergeWrites(LastWrite, Call);
                            Changed = true;
                            LastWrite = nullptr; 
                        } else {
                            LastWrite = Call;
                        }
                    } 
                    // Handle Read Hoisting
                    else if (FnName == "fread" || FnName == "read") {
                        if (hoistRead(Call, AA, DL)) {
                            Changed = true;
                        }
                        LastWrite = nullptr; 
                    } 
                    // Other function calls might have side effects, reset state
                    else {
                        LastWrite = nullptr;
                    }
                }
            }
        }

        return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
};

} // end anonymous namespace

// -----------------------------------------------------------------------------
// Pass Plugin Registration Boilerplate
// -----------------------------------------------------------------------------
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "IOOptimizationPass", LLVM_VERSION_STRING,
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "io-opt") {
                        FPM.addPass(IOOptimizationPass());
                        return true;
                    }
                    return false;
                });
        }};
}
