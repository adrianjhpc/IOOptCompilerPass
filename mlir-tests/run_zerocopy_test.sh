#!/bin/bash
set -e

CLANG=$1
IO_OPT=$2
TEST_FILE=$3

echo "[*] Compiling $TEST_FILE to ClangIR..."
$CLANG -fclangir -emit-cir $TEST_FILE -o zerocopy_test_cir.mlir

echo "[*] Running Auto-Zero-Copy Pass..."
$IO_OPT zerocopy_test_cir.mlir --allow-unregistered-dialect --io-zero-copy-promotion -o zerocopy_test_opt.mlir

echo "==========================================================="
echo " Test 1: Verifying 'proxy_clean' (Positive Test)"
echo "==========================================================="
sed -n '/@proxy_clean/,/^ *}/p' zerocopy_test_opt.mlir > clean_func.mlir

if grep -q "sendfile" clean_func.mlir && ! grep -q "write(" clean_func.mlir; then
    echo -e "[PASS] proxy_clean successfully optimized to zero-copy sendfile!"
else
    echo -e "[FAIL] proxy_clean was not optimized correctly."
    exit 1
fi

echo "==========================================================="
echo " Test 2: Verifying 'proxy_mutated' (Negative Test)"
echo "==========================================================="
sed -n '/@proxy_mutated/,/^ *}/p' zerocopy_test_opt.mlir > mut_func.mlir

if ! grep -q "sendfile" mut_func.mlir && grep -q "write(" mut_func.mlir; then
    echo -e "[PASS] proxy_mutated safely bypassed! Memory hazard was detected."
else
    echo -e "[FAIL] proxy_mutated was unsafely optimized!"
    exit 1
fi

echo "==========================================================="
echo " Test 3: Verifying 'proxy_size_mismatch' (Negative Test)"
echo "==========================================================="
sed -n '/@proxy_size_mismatch/,/^ *}/p' zerocopy_test_opt.mlir > size_func.mlir

if ! grep -q "sendfile" size_func.mlir && grep -q "write(" size_func.mlir; then
    echo -e "[PASS] proxy_size_mismatch safely bypassed! Size mismatch detected."
else
    echo -e "[FAIL] proxy_size_mismatch was unsafely optimized!"
    exit 1
fi

echo "==========================================================="
echo " Test 4: Verifying 'proxy_opaque_call' (Negative Test)"
echo "==========================================================="
sed -n '/@proxy_opaque_call/,/^ *}/p' zerocopy_test_opt.mlir > opaque_func.mlir

if ! grep -q "sendfile" opaque_func.mlir && grep -q "write(" opaque_func.mlir; then
    echo -e "[PASS] proxy_opaque_call safely bypassed! Opaque call detected."
else
    echo -e "[FAIL] proxy_opaque_call was unsafely optimized!"
    exit 1
fi

rm -f zerocopy_test_cir.mlir zerocopy_test_opt.mlir clean_func.mlir mut_func.mlir size_func.mlir opaque_func.mlir
