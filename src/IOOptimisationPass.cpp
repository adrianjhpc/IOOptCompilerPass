#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"

using namespace llvm;

namespace {

// -----------------------------------------------------------------------------
// Helper 1: Check Memory Adjacency using GEPs
// -----------------------------------------------------------------------------
bool checkAdjacency(Value *Buf1, Value *Size1, Value *Buf2, const DataLayout &DL) {
    // If Buf2 is a GetElementPtr (GEP) instruction based on Buf1
    if (auto *GEP = dyn_cast<GEPOperator>(Buf2)) {
        if (GEP->getPointerOperand() == Buf1) {
            // Check if the offset of Buf2 exactly equals the size of Buf1
            APInt Offset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
            if (GEP->accumulateConstantOffset(DL, Offset)) {
                // If Size1 is a constant, we can compare them directly
                if (auto *ConstSize1 = dyn_cast<ConstantInt>(Size1)) {
                    if (Offset == ConstSize1->getValue()) {
                        return true; // Buf2 is exactly Buf1 + Size1
                    }
                }
            }
        }
    }
    return false;
}

// -----------------------------------------------------------------------------
// Helper 2: Alias Analysis & Interference Logic
// -----------------------------------------------------------------------------
bool isSafeToAmalgamate(CallInst *Call1, CallInst *Call2, AAResults &AA, const DataLayout &DL) {
    if (Call1->getCalledFunction() != Call2->getCalledFunction()) return false;
    
    // Safety check: If the original C code uses the return value of fwrite 
    // (e.g., `int bytes_written = fwrite(...)`), we cannot easily merge them 
    // without complex math to fake the return values. We skip if return values are used.
    if (!Call1->use_empty() || !Call2->use_empty()) return false;

    Value *FilePtr1 = Call1->getArgOperand(3);
    Value *FilePtr2 = Call2->getArgOperand(3);
    if (FilePtr1 != FilePtr2) return false;

    Value *Buf1 = Call1->getArgOperand(0);
    Value *Buf2 = Call2->getArgOperand(0);
    
    // Assuming standard fwrite signature: fwrite(ptr, size, nmemb, stream)
    // Total bytes written = size * nmemb. For simplicity, we assume size (arg 1) is 1.
    Value *WriteLen1 = Call1->getArgOperand(2); 
    
    // ADJACENCY CHECK
    if (!checkAdjacency(Buf1, WriteLen1, Buf2, DL)) {
        return false; 
    }

    // INTERFERENCE CHECK
    LocationSize Size1 = LocationSize::precise(DL.getTypeStoreSize(Buf1->getType()));
    MemoryLocation Loc1(Buf1, Size1);
    MemoryLocation Loc2(Buf2, LocationSize::precise(DL.getTypeStoreSize(Buf2->getType())));

    for (Instruction *I = Call1->getNextNode(); I != Call2; I = I->getNextNode()) {
        if (!I) return false; // Call2 is not in the same basic block
        if (!I->mayReadOrWriteMemory()) continue;

        if (isModOrRefSet(AA.getModRefInfo(I, Loc1))) return false;
        if (isModOrRefSet(AA.getModRefInfo(I, Loc2))) return false;
    }

    return true; 
}

// -----------------------------------------------------------------------------
// Helper 3: The IR Builder Transformation
// -----------------------------------------------------------------------------
void mergeFWrites(CallInst *Call1, CallInst *Call2) {
    // Set up the builder to insert instructions right before Call2
    IRBuilder<> Builder(Call2);

    // Calculate the new length: Len1 + Len2
    Value *Len1 = Call1->getArgOperand(2);
    Value *Len2 = Call2->getArgOperand(2);
    Value *NewLen = Builder.CreateAdd(Len1, Len2, "merged.fwrite.len");

    // Construct the new arguments: {Buf1, Size (usually 1), NewLen, FilePtr}
    Value *Args[] = {
        Call1->getArgOperand(0), 
        Call1->getArgOperand(1), 
        NewLen, 
        Call1->getArgOperand(3)
    };

    // Create the new fwrite instruction
    Builder.CreateCall(Call1->getCalledFunction(), Args);

    // Erase the old, individual calls from the basic block
    Call1->eraseFromParent();
    Call2->eraseFromParent();
    
    errs() << "Successfully amalgamated two fwrite calls!\n";
}

// -----------------------------------------------------------------------------
// Helper 4: Hoist Read Operations
// -----------------------------------------------------------------------------
bool hoistRead(CallInst *ReadCall, AAResults &AA, const DataLayout &DL) {
    Value *DestBuf = ReadCall->getArgOperand(0);
    LocationSize ReadSize = LocationSize::precise(DL.getTypeStoreSize(DestBuf->getType()));
    MemoryLocation DestLoc(DestBuf, ReadSize);

    Instruction *InsertPoint = ReadCall;
    Instruction *Prev = ReadCall->getPrevNode();
    
    while (Prev) {
        if (Prev->isTerminator() || isa<PHINode>(Prev)) break;

        if (Prev->mayReadOrWriteMemory()) {
            ModRefInfo MR = AA.getModRefInfo(Prev, DestLoc);
            // If the previous instruction interferes with our buffer, we must stop here
            if (isModOrRefSet(MR)) break;
            
            // We also must stop if the previous instruction modifies the File Pointer
            Value *FilePtr = ReadCall->getArgOperand(3);
            MemoryLocation FilePtrLoc(FilePtr, LocationSize::beforeOrAfterPointer());
            if (isModSet(AA.getModRefInfo(Prev, FilePtrLoc))) break;
        }

        InsertPoint = Prev;
        Prev = Prev->getPrevNode();
    }

    if (InsertPoint != ReadCall) {
        ReadCall->moveBefore(InsertPoint->getIterator());
        errs() << "Successfully hoisted a read operation!\n";
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
    CallInst *LastFWrite = nullptr;

    for (Instruction &I : llvm::make_early_inc_range(BB)) {
        if (auto *Call = dyn_cast<CallInst>(&I)) {
            Function *CalledFn = Call->getCalledFunction();
            if (!CalledFn || !CalledFn->hasName()) continue;

            StringRef FnName = CalledFn->getName();

            // 1. Handle fwrite Amalgamation
            if (FnName == "fwrite") {
                if (LastFWrite && isSafeToAmalgamate(LastFWrite, Call, AA, DL)) {
                    mergeFWrites(LastFWrite, Call);
                    Changed = true;
                    LastFWrite = nullptr; 
                } else {
                    LastFWrite = Call;
                }
            } 
            // 2. Handle fread/read Hoisting
            else if (FnName == "fread" || FnName == "read") {
                if (hoistRead(Call, AA, DL)) {
                    Changed = true;
                }
                // Reset fwrite state since a read might intervene
                LastFWrite = nullptr; 
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

