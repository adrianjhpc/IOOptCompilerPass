#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse

TOOLCHAIN_DIR = "../build/mlir-src/"
LIB_DIR = "../build/llvm-src/"

# Define both compilers
CLANG_C = "clang"
CLANG_CXX = "clang++"

IO_OPT = os.path.join(TOOLCHAIN_DIR, "io-opt")
CIR_TRANSLATE = "cir-translate"
OPT = "opt"
LLVM_LINK = "llvm-link"
LLVM_AS = "llvm-as"

LLVM_PLUGIN = os.path.join(LIB_DIR, "IOOpt.so")

def run_cmd(cmd, step_name):
    """Helper to run a command and handle errors."""
    print(f"[*] {step_name}...")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"[!] Error during: {step_name}")
        sys.exit(result.returncode)

def get_compiler_for_file(filename):
    """Returns clang++ for C++ files, and clang for C files."""
    if filename.endswith(('.cpp', '.cxx', '.cc', '.hpp')):
        return CLANG_CXX
    return CLANG_C

def compile_to_bitcode(source_file, output_bc, target_triple, flags, disable_mlir):
    """Stage 1: Source -> Bitcode (Either via Fast-Path Clang or Custom MLIR Pipeline)"""
    base = os.path.splitext(source_file)[0]
    clang_target_flag = [f"--target={target_triple}"] if target_triple else []
    
    # Auto-select the frontend compiler based on the file extension
    compiler = get_compiler_for_file(source_file)

    # --- THE FAST PATH ---
    if disable_mlir:
        run_cmd([compiler, "-emit-llvm", "-c", source_file, "-o", output_bc] + clang_target_flag + flags,
                f"Fast Path ({compiler} directly to LLVM Bitcode) for {source_file}")
        return

    # --- THE MLIR PIPELINE ---
    cir_mlir_file = f"{base}_cir.mlir"
    llvm_dialect_clean_file = f"{base}_llvm_clean.mlir"
    ll_file = f"{base}.ll"
    io_opt_target_flag = [f"--mtriple={target_triple}"] if target_triple else []

    # 1. Frontend: C/C++ to ClangIR
    run_cmd([compiler, "-fclangir", "-emit-cir", source_file, "-o", cir_mlir_file] + clang_target_flag + flags,
            f"Frontend ({compiler} to CIR) for {source_file}")

    # 2. Build the End-to-End MLIR pipeline
    io_opt_cmd = [
        IO_OPT, cir_mlir_file,
        "--allow-unregistered-dialect",
        "--io-loop-batching", 
        "--convert-io-to-llvm",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-index-to-llvm",
        "--finalize-memref-to-llvm",
        "--reconcile-unrealized-casts",
        "--verify-each=false",
        "--cir-to-llvm-inhouse",
        "--remove-io-cast",
        "--reconcile-unrealized-casts",
        "-o", llvm_dialect_clean_file
    ] + io_opt_target_flag

    run_cmd(io_opt_cmd, f"End-to-End MLIR Compilation (MLIR Opt: ON)")

    # 3. Translate: Pure LLVM Dialect to LLVM IR Text
    run_cmd([CIR_TRANSLATE, llvm_dialect_clean_file, "--cir-to-llvmir", f"--target={target_triple}", "-o", ll_file],
            f"Translation to LLVM IR")
            
    # 4. Assemble: LLVM IR Text to LLVM Bitcode
    run_cmd([LLVM_AS, ll_file, "-o", output_bc],
            f"Assembling to Bitcode ({output_bc})")

    # Cleanup intermediate files
    for f in [cir_mlir_file, llvm_dialect_clean_file, ll_file]:
        if os.path.exists(f): os.remove(f)

def link_with_lto(input_bcs, output_bin, target_triple, flags, disable_llvm, requires_cxx_linker):
    """Stage 2: Link Bitcode -> LTO LLVM Pass -> Executable"""
    merged_bc = "merged.bc"
    lto_opt_bc = "merged_opt.bc"

    clang_target_flag = [f"--target={target_triple}"] if target_triple else []

    run_cmd([LLVM_LINK] + input_bcs + ["-o", merged_bc],
            "Merging Bitcode (llvm-link)")

    if disable_llvm:
        opt_cmd = [OPT, "-passes=default<O1>", merged_bc, "-o", lto_opt_bc]
    else:
        opt_cmd = [OPT, "-load-pass-plugin", LLVM_PLUGIN,
                 "-passes=io-lto-merge,default<O1>",
                 merged_bc, "-o", lto_opt_bc]

    run_cmd(opt_cmd, f"LLVM Link-Time Optimisation (LLVM Opt: {'OFF' if disable_llvm else 'ON'})")

    # Final Code Generation & Linking
    # Use clang++ if any C++ files were in the project, otherwise use clang
    linker = CLANG_CXX if requires_cxx_linker else CLANG_C
    run_cmd([linker, lto_opt_bc, "-o", output_bin] + clang_target_flag + flags,
            f"Final Code Generation & Linking (using {linker})")

    for f in [merged_bc, lto_opt_bc]:
        if os.path.exists(f): os.remove(f)

def main():
    parser = argparse.ArgumentParser(description="IO Custom Compiler Harness")
    parser.add_argument("-c", action="store_true", help="Compile and assemble, but do not link")
    parser.add_argument("-o", dest="output", help="Place the output into <file>")
    parser.add_argument("--target", dest="target", help="Generate code for the given target triple")
    
    parser.add_argument("--disable-mlir", action="store_true", help="Skip the MLIR loop batching pass")
    parser.add_argument("--disable-llvm", action="store_true", help="Skip the LLVM LTO hoisting pass")
    
    parser.add_argument("inputs", nargs="+", help="Input files")
 
    args, unknown_flags = parser.parse_known_args()

    # --- Auto-Detect Host Triple ---
    target_triple = args.target
    if not target_triple:
        try:
            res = subprocess.run([CLANG_C, "-print-target-triple"], capture_output=True, text=True, check=True)
            target_triple = res.stdout.strip()
            print(f"[*] Auto-detected host target: {target_triple}")
        except Exception:
            target_triple = "x86_64-unknown-linux-gnu"
    # ------------------------------------

    sources = [f for f in args.inputs if f.endswith(('.c', '.cpp', '.cxx', '.cc'))]
    objects = [f for f in args.inputs if f.endswith(('.o', '.bc'))]

    # Check if we need the C++ linker (if any source is a C++ file)
    requires_cxx_linker = any(src.endswith(('.cpp', '.cxx', '.cc')) for src in sources)

    if args.c:
        # Compile only mode
        for src in sources:
            out_file = args.output if args.output else f"{os.path.splitext(src)[0]}.bc"
            compile_to_bitcode(src, out_file, target_triple, unknown_flags, args.disable_mlir)
    else:
        # Compile and link mode
        bcs_to_link = objects.copy()
        for src in sources:
            bc_file = f"{os.path.splitext(src)[0]}.bc"
            compile_to_bitcode(src, bc_file, target_triple, unknown_flags, args.disable_mlir)
            bcs_to_link.append(bc_file)

        out_bin = args.output if args.output else "a.out"
        link_with_lto(bcs_to_link, out_bin, target_triple, unknown_flags, args.disable_llvm, requires_cxx_linker)

if __name__ == "__main__":
    main()
