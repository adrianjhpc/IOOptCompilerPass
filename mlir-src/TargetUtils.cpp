#include "TargetUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

// Define the command-line option
static llvm::cl::opt<std::string> targetTriple(
    "mtriple", llvm::cl::desc("Override target triple for the module"),
    llvm::cl::init(""));

namespace mlir {
namespace io {

void bootstrapTargetInfo(ModuleOp module) {
    // Guard to ensure that if the module already has the attributes, we don't need to re-do it
    if (module->hasAttr("llvm.target_triple") && module->hasAttr("llvm.data_layout")) {
        return;
    }

    // Determine the Triple (User Flag > Host Detect)
    std::string triple = targetTriple.empty() 
                         ? llvm::sys::getDefaultTargetTriple() 
                         : targetTriple;

    // Initialize all targets
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();

    // Lookup the target
    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        llvm::errs() << "[IOOpt] Warning: Could not find target for " << triple << ": " << error << "\n";
        return;
    }

    // Create a temporary TargetMachine to get the real Data Layout
    llvm::TargetOptions opt;
    auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(llvm::Triple(triple), "generic", "", opt, llvm::Reloc::PIC_));

    std::string layout = targetMachine->createDataLayout().getStringRepresentation();

    // Stamp the attributes
    MLIRContext *ctx = module.getContext();
    module->setAttr("llvm.target_triple", StringAttr::get(ctx, triple));
    module->setAttr("llvm.data_layout", StringAttr::get(ctx, layout));

    llvm::errs() << "[IOOpt] Targeting: " << triple << "\n";
}

} // namespace io
} // namespace mlir
