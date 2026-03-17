#!/usr/bin/env python3
import sys
import os
import subprocess
import shutil

HARNESS = "../harness-src/ioopt.py"
BASELINE_APP = "./test_app_orig"
TEST_APP = "./test_app"
OUTPUT_FILE = "output_test.txt"
BASELINE_OUTPUT = "output_orig.txt"

# Terminal colors for professional output
C_GREEN = '\033[92m'
C_RED = '\033[91m'
C_CYAN = '\033[96m'
C_RESET = '\033[0m'

def run_cmd(cmd, suppress_output=False):
    """Runs a shell command and optionally returns its stderr."""
    res = subprocess.run(cmd, text=True, capture_output=True)
    if res.returncode != 0:
        print(f"{C_RED}[FAIL] Command failed: {' '.join(cmd)}{C_RESET}")
        print(res.stderr)
        sys.exit(res.returncode)
    
    if not suppress_output and res.stdout:
        print(res.stdout.strip())
        
    return res.stderr # strace prints to stderr!

def print_side_by_side(strace_orig, strace_test):
    """Parses and prints two strace -c outputs side-by-side."""
    orig_lines = [line for line in strace_orig.strip().split('\n') if line.strip()]
    test_lines = [line for line in strace_test.strip().split('\n') if line.strip()]
    
    max_len = max(len(orig_lines), len(test_lines))
    orig_lines += [''] * (max_len - len(orig_lines))
    test_lines += [''] * (max_len - len(test_lines))
    
    print(f"\n{C_CYAN}{'BASELINE (test_app_orig)':<45} | {'OPTIMISED (test_app)'}{C_RESET}")
    print("-" * 95)
    for o, t in zip(orig_lines, test_lines):
        print(f"{o[:44]:<45} | {t}")
    print("\n")

def run_test_scenario(name, flags):
    print(f"{C_CYAN}==========================================================={C_RESET}")
    print(f"{C_CYAN} SCENARIO: {name}{C_RESET}")
    print(f"{C_CYAN}==========================================================={C_RESET}")
    
    # 1. Compile with the harness
    print("[*] Compiling through IOOpt Harness...")
    cmd = ["python3", HARNESS, "end_to_end_main.cpp", "end_to_end_lib.cpp", "-o", TEST_APP] + flags
    run_cmd(cmd, suppress_output=True)
    
    # 2. Run the test app to generate the output file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    run_cmd([TEST_APP], suppress_output=True)
    
    # 3. Verify correctness against the baseline output
    print("[*] Verifying file contents...")
    with open(OUTPUT_FILE, 'r') as f:
        test_out = f.read()
    with open(BASELINE_OUTPUT, 'r') as f:
        orig_out = f.read()
        
    if test_out == orig_out:
        print(f"{C_GREEN}[PASS] Binary executed perfectly and output matches baseline!{C_RESET}")
    else:
        print(f"{C_RED}[FAIL] Output does not match the baseline!{C_RESET}")
        sys.exit(1)
        
    # 4. Run strace -c on both binaries and compare
    print("[*] Collecting strace statistics...")
    strace_orig = run_cmd(["strace", "-c", BASELINE_APP], suppress_output=True)
    strace_test = run_cmd(["strace", "-c", TEST_APP], suppress_output=True)
    
    print_side_by_side(strace_orig, strace_test)


def main():
    print(f"{C_CYAN}==========================================================={C_RESET}")
    print(f"{C_CYAN} Building the Baseline Application{C_RESET}")
    print(f"{C_CYAN}==========================================================={C_RESET}")
    
    # Compile baseline
    run_cmd(["clang++", "-o", BASELINE_APP, "end_to_end_main.cpp", "end_to_end_lib.cpp", "-O3", "-flto"], suppress_output=True)
    
    # Generate baseline output data
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    run_cmd([BASELINE_APP], suppress_output=True)
    
    if not os.path.exists(OUTPUT_FILE):
        print(f"{C_RED}[FAIL] Baseline app failed to generate {OUTPUT_FILE}{C_RESET}")
        sys.exit(1)
        
    shutil.copy(OUTPUT_FILE, BASELINE_OUTPUT)
    print(f"{C_GREEN}[*] Baseline compiled and expected output captured.{C_RESET}\n")

    # --- Run the Scenarios ---
    
    # Scenario 1: Both Passes Enabled
    run_test_scenario(
        name="Both Passes Enabled (MLIR Batching + LLVM Hoisting)", 
        flags=[]
    )
    
    # Scenario 2: MLIR Only
    run_test_scenario(
        name="MLIR Batching Only (LLVM Disabled)", 
        flags=["--disable-llvm"]
    )
    
    # Scenario 3: LLVM Only
    run_test_scenario(
        name="LLVM Hoisting Only (MLIR Disabled)", 
        flags=["--disable-mlir"]
    )
    
    print(f"{C_GREEN}==========================================================={C_RESET}")
    print(f"{C_GREEN} All End-to-End Tests Passed Successfully!{C_RESET}")
    print(f"{C_GREEN}==========================================================={C_RESET}")

    # Cleanup
    for f in [OUTPUT_FILE, BASELINE_OUTPUT]:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    main()
