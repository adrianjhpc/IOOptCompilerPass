#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse
import re

TOOLCHAIN_DIR = "../build/mlir-src/"
LIB_DIR = "../build/llvm-src/"

CLANG = "clang++"
IO_OPT = os.path.join(TOOLCHAIN_DIR, "io-opt")
CIR_TRANSLATE = "cir-translate"
OPT = "opt"
CIR_OPT = "cir-opt"
LLVM_LINK = "llvm-link"
LLVM_AS = "llvm-as"

LLVM_PLUGIN = os.path.join(LIB_DIR, "IOOpt.so")
RUNTIME_LIB = "-lIORuntime"

def run_cmd(cmd, step_name):
    """Helper to run a command and handle errors."""
    print(f"[*] {step_name}...")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"[!] Error during: {step_name}")
        sys.exit(result.returncode)

def compile_to_bitcode(source_file, output_bc, flags):
    """Stage 1: Source -> CIR -> Std MLIR -> IO Opt -> LLVM Dialect -> LLVM IR -> LLVM Bitcode"""
    base = os.path.splitext(source_file)[0]
    cir_mlir_file = f"{base}_cir.mlir"
    llvm_dialect_clean_file = f"{base}_llvm_clean.mlir"
    ll_file = f"{base}.ll"

    # 1. Frontend: C++ to ClangIR
    run_cmd([CLANG, "-fclangir", "-emit-cir", source_file, "-o", cir_mlir_file] + flags,
            f"Frontend (Source code to CIR) for {source_file}")

    # 2. The Mega-Pipeline: End-to-End lowering in a single binary!
    run_cmd([IO_OPT, cir_mlir_file,
             "--allow-unregistered-dialect",
             "--io-loop-batching",
             "--convert-io-to-llvm",
             "--convert-scf-to-cf",
             "--convert-cf-to-llvm",
             "--convert-arith-to-llvm",
             "--convert-index-to-llvm",
             "--finalize-memref-to-llvm",
             "--reconcile-unrealized-casts",
             
             # --- The In-Memory Bypass ---
             "--verify-each=false",          # Turn off the global verifier
             "--cir-to-llvm-inhouse",        # 1. Let ClangIR do its thing first
             "--remove-io-cast",             # 2. We clean up our casts and emit ptrtoint
             "--reconcile-unrealized-casts", # 3. Garbage collect the temporary casts
             
             "-o", llvm_dialect_clean_file],
            f"End-to-End MLIR Compilation")

    # 3. Translate: Pure LLVM Dialect to LLVM IR Text
    run_cmd([CIR_TRANSLATE, llvm_dialect_clean_file, "--cir-to-llvmir", "-o", ll_file],
            f"Translation to LLVM IR")
            
    # 4. Assemble: LLVM IR Text to LLVM Bitcode (This creates the actual .bc file!)
    run_cmd([LLVM_AS, ll_file, "-o", output_bc],
            f"Assembling to Bitcode ({output_bc})")

    # Cleanup intermediate files
    for f in [cir_mlir_file, llvm_dialect_clean_file, ll_file, ]:
        if os.path.exists(f): os.remove(f)

def link_with_lto(input_bcs, output_bin, flags):
    """Stage 2: Link Bitcode -> LTO LLVM Pass -> Executable"""
    merged_bc = "merged.bc"
    lto_opt_bc = "merged_opt.bc"

    #  Merge all bitcode files together (The LTO Link step)
    run_cmd([LLVM_LINK] + input_bcs + ["-o", merged_bc],
            "Merging Bitcode (llvm-link)")

    # Run the LLVM Pass across the entire merged program
    # This allows interprocedural LLVM optimisations across different .cpp files
    run_cmd([OPT, "-load-pass-plugin", LLVM_PLUGIN, 
             "-passes=io-lto-merge,default<O1>", 
             merged_bc, "-o", lto_opt_bc],      
            "LLVM Link-Time Optimisation (LTO)")

    # Final Machine Code Generation & Native Linking
    run_cmd([CLANG, lto_opt_bc, "-o", output_bin] + flags,
            "Final Code Generation & Linking")

    # Cleanup
    for f in [merged_bc, lto_opt_bc]:
        if os.path.exists(f): os.remove(f)

def main():
    parser = argparse.ArgumentParser(description="IO Custom Compiler Harness")
    parser.add_argument("-c", action="store_true", help="Compile and assemble, but do not link")
    parser.add_argument("-o", dest="output", help="Place the output into <file>")
    parser.add_argument("inputs", nargs="+", help="Input files")
    
    # Capture unknown arguments to pass down to clang (like -O3, -g, -I...)
    args, unknown_flags = parser.parse_known_args()

    sources = [f for f in args.inputs if f.endswith(('.c', '.cpp', '.cxx'))]
    objects = [f for f in args.inputs if f.endswith(('.o', '.bc'))]

    if args.c:
        # Compile only mode
        for src in sources:
            out_file = args.output if args.output else f"{os.path.splitext(src)[0]}.bc"
            compile_to_bitcode(src, out_file, unknown_flags)
    else:
        # Compile and link mode
        bcs_to_link = objects.copy()
        for src in sources:
            bc_file = f"{os.path.splitext(src)[0]}.bc"
            compile_to_bitcode(src, bc_file, unknown_flags)
            bcs_to_link.append(bc_file)
        
        out_bin = args.output if args.output else "a.out"
        link_with_lto(bcs_to_link, out_bin, unknown_flags)

if __name__ == "__main__":
    main()
