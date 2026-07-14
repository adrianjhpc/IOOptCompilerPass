#!/usr/bin/env bash
set -euo pipefail

: "${PLUGIN:=/home/adrianj/IOOptCompilerPass/build/llvm-src/IOOpt.so}"
: "${OPT:=/data/llvm-install/bin/opt}"          # 21.1.8: the proven plugin host
WORK=${WORK:-/tmp/ioopt-exec}
DIR=$(dirname "$0")
SRC=${SRC:-$DIR/vector_loop_collapse_harness.c}
mkdir -p "$WORK"

# --- Derive the plugin's LLVM major from opt, then find a MATCHING clang. ---
MAJ="$("$OPT" --version | sed -n 's/.*version \([0-9]*\).*/\1/p' | head -1)"
echo "Plugin/opt LLVM major = $MAJ"

find_cc() {
  for c in "$(dirname "$OPT")/clang" "clang-$MAJ" "/usr/bin/clang-$MAJ" clang; do
    command -v "$c" >/dev/null 2>&1 || continue
    cm="$("$c" --version | sed -n 's/.*version \([0-9]*\).*/\1/p' | head -1)"
    [ "$cm" = "$MAJ" ] && { echo "$c"; return 0; }
  done
  return 1
}
CC="$(find_cc)" || {
  echo "FATAL: plugin/opt are LLVM $MAJ but no clang-$MAJ found."
  echo "Building an executable would mix IR versions (the source of every"
  echo "'Unknown attribute kind' failure this session). Install/point to a"
  echo "clang $MAJ, or rebuild the plugin against your clang's version."
  exit 1
}
echo "Using matched CC = $CC ($("$CC" --version | head -1))"

PASSES='loop-simplify,lcssa,io-opt'                     # canonicalize, then io-opt
FLAGS='-io-opt-loop-hoist-dynamic-trips -io-opt-loop-vectored'  # PLAIN, to opt

# --- Baseline: plain -O2, no plugin. ---
"$CC" -O2 -fno-inline "$SRC" -o "$WORK/base"

# --- Optimized: clang -> textual IR -> opt(plugin) -> clang backend. ---
"$CC" -O2 -fno-inline -emit-llvm -S "$SRC" -o "$WORK/pre.ll"
IO_ENABLE_LOGGING=1 "$OPT" -load-pass-plugin="$PLUGIN" -passes="$PASSES" \
    $FLAGS "$WORK/pre.ll" -S -o "$WORK/opt.ll" 2> "$WORK/fire.log"
"$CC" -O2 -fno-inline "$WORK/opt.ll" -o "$WORK/opt"

# --- Confirm the transforms actually fired (else "identical" is vacuous). ---
echo "=== IOOpt activity ==="; grep '\[IOOpt\]' "$WORK/fire.log" || true
need() { grep -q "$1" "$WORK/fire.log" || { echo "FAIL(fire): '$1' not seen"; exit 1; }; }
need "Hoisted DYNAMIC WRITE"     # case A
need "Hoisted DYNAMIC READ"      # case B
need "DVLC collapsed"     # case D
echo "All three transforms fired."

# --- Run baseline vs optimized and diff outputs. ---
pass=0; fail=0
check() {
  if cmp -s "$WORK/base.$1" "$WORK/opt.$1"; then
    echo "PASS  $1 ($(stat -c%s "$WORK/base.$1") bytes, identical)"; pass=$((pass+1))
  else echo "FAIL  $1 (outputs differ)"; fail=$((fail+1)); fi
}
N=1000
"$WORK/base" A $N "$WORK/base.A"; "$WORK/opt" A $N "$WORK/opt.A"; check A
"$WORK/base" A 0  "$WORK/base.C"; "$WORK/opt" A 0  "$WORK/opt.C"; check C
"$WORK/base" A $N "$WORK/ref.in"
"$WORK/base" B $N "$WORK/base.B" "$WORK/ref.in"
"$WORK/opt"  B $N "$WORK/opt.B"  "$WORK/ref.in"; check B
cmp -s "$WORK/opt.B" "$WORK/ref.in" && echo "PASS  B==input" \
     || { echo "FAIL  B!=input"; fail=$((fail+1)); }
"$WORK/base" D "$WORK/base.D"; "$WORK/opt" D "$WORK/opt.D"; check D

echo "=================================================="
echo "PASS=$pass FAIL=$fail"
[ "$fail" -eq 0 ] && echo "ALL EXECUTION TESTS PASSED" \
                  || { echo "EXECUTION TESTS FAILED"; exit 1; }

