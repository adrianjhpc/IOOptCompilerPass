import subprocess
import re
import sys
import argparse

def count_syscalls(binary_path):
    print(f"--- Running {binary_path} ---")
    # We use strace -c to get a summary count of system calls
    cmd = ["strace", "-c", binary_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        # strace outputs its statistics to stderr
        stats = result.stderr

        # Regex to capture the 'calls' column for write and writev
        # Matches lines ending in 'write' or 'writev' and extracts the call count
        write_match = re.search(r'^\s*[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+(\d+).*?\bwrite$', stats, re.MULTILINE)
        writev_match = re.search(r'^\s*[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+(\d+).*?\bwritev$', stats, re.MULTILINE)

        return {
            "write": int(write_match.group(1)) if write_match else 0,
            "writev": int(writev_match.group(1)) if writev_match else 0
        }
    except Exception as e:
        print(f"Error running strace on {binary_path}: {e}")
        return None

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Compare I/O syscalls between two executables.")
    parser.add_argument("standard_exe", help="Path to the unoptimized executable (e.g., ./bench_standard)")
    parser.add_argument("fast_exe", help="Path to the optimized executable (e.g., ./bench_fast)")
    args = parser.parse_args()

    std_data = count_syscalls(args.standard_exe)
    fast_data = count_syscalls(args.fast_exe)

    if std_data is not None and fast_data is not None:
        std_total = std_data['write'] + std_data['writev']
        fast_total = fast_data['write'] + fast_data['writev']

        print("\n" + "="*40)
        print(f"{'Syscall':<15} | {'Standard':<10} | {'Optimized':<10}")
        print("-" * 40)
        print(f"{'write':<15} | {std_data['write']:<10} | {fast_data['write']:<10}")
        print(f"{'writev':<15} | {std_data['writev']:<10} | {fast_data['writev']:<10}")
        print("-" * 40)
        print(f"{'TOTAL I/O':<15} | {std_total:<10} | {fast_total:<10}")
        print("="*40)

        if fast_total < std_total:
            reduction = ((std_total - fast_total) / std_total) * 100
            print(f"SUCCESS: Your pass reduced I/O syscalls by {reduction:.2f}%")
        elif fast_data['writev'] > 0 and std_data['writev'] == 0 and fast_total == std_total:
            print("SUCCESS: Converted writes to writev (though total count remained the same).")
        else:
            print("FAILURE: No reduction in syscalls detected.")
        print("="*40)

if __name__ == "__main__":
    main()
