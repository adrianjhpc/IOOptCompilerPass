#!/usr/bin/env python3
import sys
import os
import subprocess
import argparse

TOOLCHAIN_DIR = "/home/adrianj/IOOptCompilerPass/build/mlir-src/"
LIB_DIR = "/home/adrianj/IOOptCompilerPass/build/llvm-src/"

# Define standard toolchain
CLANG_C = "clang"
CLANG_CXX = "clang++"
LLVM_AR = "llvm-ar" 

IO_OPT = os.path.join(TOOLCHAIN_DIR, "io-opt")
CIR_TRANSLATE = "cir-translate"
OPT = "opt"
LLVM_LINK = "llvm-link"
LLVM_AS = "llvm-as"

LLVM_PLUGIN = os.path.join(LIB_DIR, "IOOpt.so")

def run_cmd(cmd, step_name, allow_failure=False):
    """Helper to run a command and handle errors."""
    print(f"[*] {step_name}...")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        if allow_failure:
            print(f"[!] Warning: {step_name} failed. Auto-falling back to standard compiler...")
            return False
        else:
            print(f"[!] Error during: {step_name}")
            sys.exit(result.returncode)
    return True

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

    # --- THE EXPLICIT FAST PATH (User requested) ---
    if disable_mlir:
        run_cmd([compiler, "-emit-llvm", "-c", source_file, "-o", output_bc] + clang_target_flag + flags,
                f"Fast Path ({compiler} directly to LLVM Bitcode) for {source_file}")
        return

    # --- THE MLIR PIPELINE ---
    cir_mlir_file = f"{base}_cir.mlir"
    llvm_dialect_clean_file = f"{base}_llvm_clean.mlir"
    ll_file = f"{base}.ll"
    io_opt_target_flag = [f"--mtriple={target_triple}"] if target_triple else []

    def cleanup_intermediates():
        for f in [cir_mlir_file, llvm_dialect_clean_file, ll_file]:
            if os.path.exists(f): os.remove(f)

    # 1. Frontend: C/C++ to ClangIR (NOW WITH FALLBACK)
    frontend_success = run_cmd(
        [compiler, "-fclangir", "-emit-cir", source_file, "-o", cir_mlir_file] + clang_target_flag + flags,
        f"Frontend ({compiler} to CIR) for {source_file}",
        allow_failure=True
    )

    if not frontend_success:
        print(f"[*] ClangIR Frontend failed. Auto-falling back to standard Clang for {source_file}...")
        run_cmd([compiler, "-emit-llvm", "-c", source_file, "-o", output_bc] + clang_target_flag + flags,
                f"Fallback Fast Path ({compiler} directly to LLVM Bitcode)")
        cleanup_intermediates()
        return

    # 2. Build the End-to-End MLIR pipeline
    io_opt_cmd = [
        IO_OPT, cir_mlir_file,
        "--allow-unregistered-dialect",
        "--io-zero-copy-promotion",
        "--io-loop-batching", 
        "--cir-to-llvm-inhouse",
        "--convert-io-to-llvm",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-index-to-llvm",
        "--finalize-memref-to-llvm",
        "--reconcile-unrealized-casts",
        "--verify-each=false",
        "--remove-io-cast",
        "--reconcile-unrealized-casts",
        "-o", llvm_dialect_clean_file
    ] + io_opt_target_flag

    # 3. Execute MLIR Passes (WITH FALLBACK)
    mlir_success = run_cmd(io_opt_cmd, f"End-to-End MLIR Compilation", allow_failure=True)
    
    if not mlir_success:
        print(f"[*] MLIR Passes failed. Auto-falling back to standard Clang for {source_file}...")
        run_cmd([compiler, "-emit-llvm", "-c", source_file, "-o", output_bc] + clang_target_flag + flags,
                f"Fallback Fast Path ({compiler} directly to LLVM Bitcode)")
        cleanup_intermediates()
        return

    # 4. Translate: Pure LLVM Dialect to LLVM IR Text
    run_cmd([CIR_TRANSLATE, llvm_dialect_clean_file, "--cir-to-llvmir", f"--target={target_triple}", "-o", ll_file],
            f"Translation to LLVM IR")
            
    # 5. Assemble: LLVM IR Text to LLVM Bitcode
    run_cmd([LLVM_AS, ll_file, "-o", output_bc],
            f"Assembling to Bitcode ({output_bc})")

    cleanup_intermediates()

def link_with_lto(input_bcs, output_bin, target_triple, flags, disable_llvm, requires_cxx_linker):
    """Stage 2: Link Bitcode -> LTO LLVM Pass -> Executable/Library"""
    
    # --- 1. Static Library (.a) Fast Path ---
    if output_bin.endswith('.a'):
        # For static libraries, we just archive the bitcode files. 
        # The LTO pass will run when the executable eventually links this archive.
        run_cmd([LLVM_AR, "rcs", output_bin] + input_bcs,
                f"Archiving Static Library ({output_bin})")
        return

    # --- 2. Multiple Binaries (Concurrency Fix) ---
    # Use the target binary name to create unique intermediate files. 
    # This prevents parallel 'make -j' processes from overwriting each other's bitcode!
    base_name = os.path.splitext(output_bin)[0]
    merged_bc = f"{base_name}_merged.bc"
    lto_opt_bc = f"{base_name}_merged_opt.bc"

    clang_target_flag = [f"--target={target_triple}"] if target_triple else []
    
    # --- 3. Shared Objects (.so) Fix ---
    # Ensure -shared is passed down if requested by the build system
    is_shared = "-shared" in flags or output_bin.endswith('.so')
    shared_flag = ["-shared"] if is_shared and "-shared" not in flags else []

    run_cmd([LLVM_LINK] + input_bcs + ["-o", merged_bc],
            f"Merging Bitcode (llvm-link) for {output_bin}")

    if disable_llvm:
        opt_cmd = [OPT, "-passes=default<O1>", merged_bc, "-o", lto_opt_bc]
    else:
        opt_cmd = [OPT, "-load-pass-plugin", LLVM_PLUGIN,
                 "-passes=io-lto-merge,default<O1>",
                 merged_bc, "-o", lto_opt_bc]

    run_cmd(opt_cmd, f"LLVM Link-Time Optimisation for {output_bin} (LLVM Opt: {'OFF' if disable_llvm else 'ON'})")

    # Final Code Generation & Linking
    linker = CLANG_CXX if requires_cxx_linker else CLANG_C
    
    # Force LLD and LTO so the linker can read bitcode archives (.a) 
    lto_flags = ["-fuse-ld=lld", "-flto"] 
    
    final_cmd = [linker, lto_opt_bc, "-o", output_bin] + clang_target_flag + shared_flag + lto_flags + flags
    
    run_cmd(final_cmd, f"Final Code Generation & Linking ({output_bin} using {linker})")

    # Cleanup
    for f in [merged_bc, lto_opt_bc]:
        if os.path.exists(f): os.remove(f)

def main():
    # --- 1. Manual Argument Parsing ---
    args_c = False
    args_output = None
    target_triple = None
    disable_mlir = False
    disable_llvm = False
    
    sources = []
    objects = []
    unknown_flags = []
    
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        
        if arg == "-c":
            args_c = True
        elif arg == "-o":
            if i + 1 < len(argv):
                args_output = argv[i+1]
                i += 1
        elif arg.startswith("-o"): # Handle glued args like -ofoo.o
            args_output = arg[2:]
        elif arg == "--target":
            if i + 1 < len(argv):
                target_triple = argv[i+1]
                i += 1
        elif arg.startswith("--target="):
            target_triple = arg.split("=", 1)[1]
        elif arg == "--disable-mlir":
            disable_mlir = True
        elif arg == "--disable-llvm":
            disable_llvm = True
        elif arg.startswith("-"):
            # Any unrecognized flag (like -Wall, -D, -I, -O3) goes straight through
            unknown_flags.append(arg)
        else:
            # It's a positional argument. Sort it based on extension!
            if arg.endswith(('.c', '.cpp', '.cxx', '.cc')):
                sources.append(arg)
            elif arg.endswith(('.o', '.bc', '.a')):
                objects.append(arg)
            else:
                # E.g., a path to a linker script or a bare value like "-I /usr/include"
                unknown_flags.append(arg)
        i += 1

    # --- 2. Auto-Detect Host Triple ---
    if not target_triple:
        try:
            res = subprocess.run([CLANG_C, "-print-target-triple"], capture_output=True, text=True, check=True)
            target_triple = res.stdout.strip()
            print(f"[*] Auto-detected host target: {target_triple}")
        except Exception:
            target_triple = "x86_64-unknown-linux-gnu"

    # --- 3. Route to the correct pipeline ---
    requires_cxx_linker = any(src.endswith(('.cpp', '.cxx', '.cc')) for src in sources)

    if args_c:
        # Compile only mode
        for src in sources:
            out_file = args_output if args_output else f"{os.path.splitext(src)[0]}.o"
            compile_to_bitcode(src, out_file, target_triple, unknown_flags, disable_mlir)
    else:
        # Compile and link mode
        bcs_to_link = objects.copy()
        for src in sources:
            bc_file = f"{os.path.splitext(src)[0]}.o"
            compile_to_bitcode(src, bc_file, target_triple, unknown_flags, disable_mlir)
            bcs_to_link.append(bc_file)

        out_bin = args_output if args_output else "a.out"
        link_with_lto(bcs_to_link, out_bin, target_triple, unknown_flags, disable_llvm, requires_cxx_linker)

if __name__ == "__main__":
    main()
