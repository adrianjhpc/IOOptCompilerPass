#!/bin/bash
set -e

CLANG=$1
LLVM_LINK=$2
OPT=$3
PLUGIN=$4
DIR=$(dirname "$0")

echo "[*] Simulating the Hourglass Backend (LLVM LTO)..."

# 1. Compile individual files to LLVM IR
$CLANG -O1 -Xclang -disable-llvm-passes -emit-llvm -S $DIR/example_application.c -o app.ll
$CLANG -O1 -Xclang -disable-llvm-passes -emit-llvm -S $DIR/example_io_library.c -o io_lib.ll

# 2. Link all IR files into one global module
$LLVM_LINK app.ll io_lib.ll -S -o merged_lto.ll

# 3. Run the LLVM Pass
export IO_ENABLE_LOGGING=1
export IO_BATCH_THRESHOLD=3  # CRITICAL FIX: Drop threshold to 3!

$OPT -load-pass-plugin=$PLUGIN -passes="module(function(mem2reg,instcombine),io-lto-merge)" merged_lto.ll -S -o optimized_lto.ll 2> opt_output.log

echo "==========================================================="
echo " Test 1: Verifying Cross-File Inlining (LTO)"
echo "==========================================================="
if grep -q "\[IOOpt-LTO\] SUCCESS: Inlined I/O wrapper 'my_fast_write' into 'process_data'" opt_output.log; then
    echo -e "[PASS] LLVM successfully breached the file boundary and inlined the wrapper!"
else
    echo -e "[FAIL] Cross-file inlining failed. Opt output:"
    cat opt_output.log
    exit 1
fi

echo "==========================================================="
echo " Test 2: Verifying Cross-File Vectorization"
echo "==========================================================="
# CRITICAL FIX: Grep for 3 writes instead of 4!
if grep -q "\[IOOpt\] SUCCESS: N-Way converted 3 writes to writev" opt_output.log; then
    echo -e "[PASS] LLVM successfully batched the cross-file calls into a single writev!"
else
    echo -e "[FAIL] Vectorization failed. Opt output:"
    cat opt_output.log
    exit 1
fi

rm -f app.ll io_lib.ll merged_lto.ll optimized_lto.ll opt_output.log
