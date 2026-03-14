#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse

TOOLCHAIN_DIR = "./bin"
LIB_DIR = "./lib"

CLANG = "clang++"
IO_OPT = os.path.join(TOOLCHAIN_DIR, "io-opt")
MLIR_TRANSLATE = os.path.join(TOOLCHAIN_DIR, "mlir-translate")
OPT = "opt"
LLVM_LINK = "llvm-link"

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
    """Stage 1: Source -> MLIR -> LLVM IR -> LLVM Bitcode"""
    base = os.path.splitext(source_file)[0]
    mlir_file = f"{base}.mlir"
    opt_mlir_file = f"{base}_opt.mlir"
    ll_file = f"{base}.ll"

    # Frontend: C++ to ClangIR/MLIR
    run_cmd([CLANG, "-fclangir", "-emit-mlir", source_file, "-o", mlir_file] + flags, 
            f"Frontend (Source code to MLIR) for {source_file}")

    # MLIR Pass: io-opt
    run_cmd([IO_OPT, mlir_file, "--convert-cir-to-io", "--io-batch-merge", 
             "--convert-io-to-llvm", "-o", opt_mlir_file], 
            f"MLIR Optimisation (io-opt)")

    # Translate: MLIR to LLVM IR
    run_cmd([MLIR_TRANSLATE, "--mlir-to-llvmir", opt_mlir_file, "-o", ll_file], 
            f"Translation to LLVM IR")

    # LLVM Pass (Compile-Time) & Emit Bitcode
    run_cmd([OPT, "-load-pass-plugin", LLVM_PLUGIN, "-passes=io-cleanup,default<O2>", 
             ll_file, "-o", output_bc], 
            f"LLVM Compile-Time Optimisation")

    # Cleanup intermediate files
    for f in [mlir_file, opt_mlir_file, ll_file]:
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
    run_cmd([OPT, "-load-pass-plugin", LLVM_PLUGIN, "-passes=io-lto-merge,default<O3>", 
             merged_bc, "-o", lto_opt_bc], 
            "LLVM Link-Time Optimisation (LTO)")

    # Final Machine Code Generation & Native Linking
    run_cmd([CLANG, lto_opt_bc, "-o", output_bin, f"-L{LIB_DIR}", RUNTIME_LIB] + flags, 
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
