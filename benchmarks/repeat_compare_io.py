import subprocess
import re
import argparse
import hashlib
import os
import statistics

def get_sha256(filepath):
    """Calculates the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()
    except Exception as e:
        return None

def benchmark_syscalls(binary_path, iterations):
    """Runs strace multiple times and returns statistical data on write/writev calls."""
    print(f"--- Stracing {binary_path} ({iterations} iterations) ---")
    write_counts = []
    writev_counts = []

    for i in range(iterations):
        cmd = ["strace", "-c", binary_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            stats = result.stderr

            write_match = re.search(r'^\s*[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+(\d+).*?\bwrite$', stats, re.MULTILINE)
            writev_match = re.search(r'^\s*[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+(\d+).*?\bwritev$', stats, re.MULTILINE)

            w_count = int(write_match.group(1)) if write_match else 0
            wv_count = int(writev_match.group(1)) if writev_match else 0

            write_counts.append(w_count)
            writev_counts.append(wv_count)
            
        except Exception as e:
            print(f"Error running strace on {binary_path} (iteration {i+1}): {e}")
            return None

    # Calculate statistics
    def get_stats(data):
        mean_val = statistics.mean(data)
        std_val = statistics.stdev(data) if iterations > 1 else 0.0
        return mean_val, std_val

    total_counts = [w + wv for w, wv in zip(write_counts, writev_counts)]

    return {
        "write": get_stats(write_counts),
        "writev": get_stats(writev_counts),
        "total": get_stats(total_counts)
    }

def main():
    parser = argparse.ArgumentParser(description="Compare I/O syscalls with statistical analysis and verify data integrity.")
    parser.add_argument("standard_exe", help="Path to the unoptimized executable")
    parser.add_argument("fast_exe", help="Path to the optimized executable")
    parser.add_argument("--out-file", dest="out_file", help="The name of the log file to verify", default=None)
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of times to run each benchmark (default: 10)")
    args = parser.parse_args()

    # --- Run Standard Benchmark ---
    if args.out_file and os.path.exists(args.out_file):
        os.remove(args.out_file) 
        
    std_data = benchmark_syscalls(args.standard_exe, args.iterations)
    std_hash = None
    
    if args.out_file and os.path.exists(args.out_file):
        os.rename(args.out_file, "standard.out")
        std_hash = get_sha256("standard.out")

    # --- Run Optimized Benchmark ---
    if args.out_file and os.path.exists(args.out_file):
        os.remove(args.out_file)
        
    fast_data = benchmark_syscalls(args.fast_exe, args.iterations)
    fast_hash = None
    
    if args.out_file and os.path.exists(args.out_file):
        os.rename(args.out_file, "fast.out")
        fast_hash = get_sha256("fast.out")

    # --- Print Syscall Results ---
    if std_data is not None and fast_data is not None:
        print("\n" + "="*55)
        print(f"{'Syscall':<15} | {'Standard':<15} | {'Optimized':<15}")
        print("-" * 55)
        
        # Display as "Mean ± StdDev"
        w_std_str = f"{std_data['write'][0]:.1f} ± {std_data['write'][1]:.1f}"
        w_fst_str = f"{fast_data['write'][0]:.1f} ± {fast_data['write'][1]:.1f}"
        print(f"{'write':<15} | {w_std_str:<15} | {w_fst_str:<15}")

        wv_std_str = f"{std_data['writev'][0]:.1f} ± {std_data['writev'][1]:.1f}"
        wv_fst_str = f"{fast_data['writev'][0]:.1f} ± {fast_data['writev'][1]:.1f}"
        print(f"{'writev':<15} | {wv_std_str:<15} | {wv_fst_str:<15}")
        
        print("-" * 55)
        
        tot_std_str = f"{std_data['total'][0]:.1f} ± {std_data['total'][1]:.1f}"
        tot_fst_str = f"{fast_data['total'][0]:.1f} ± {fast_data['total'][1]:.1f}"
        print(f"{'TOTAL I/O':<15} | {tot_std_str:<15} | {tot_fst_str:<15}")
        print("="*55)

        std_total_mean = std_data['total'][0]
        fast_total_mean = fast_data['total'][0]

        if fast_total_mean < std_total_mean:
            reduction = ((std_total_mean - fast_total_mean) / std_total_mean) * 100
            print(f"SUCCESS: Your pass reduced I/O syscalls by {reduction:.2f}% on average")
        elif fast_data['writev'][0] > 0 and std_data['writev'][0] == 0:
            print("SUCCESS: Converted writes to writev!")
        else:
            print("FAILURE: No I/O optimization detected.")
        print("="*55)

    # --- Print Integrity Results ---
    if args.out_file:
        print("\n" + "="*55)
        print("DATA INTEGRITY CHECK (From final iteration):")
        print(f"Standard Hash : {std_hash or 'FILE NOT FOUND'}")
        print(f"Optimized Hash: {fast_hash or 'FILE NOT FOUND'}")
        print("-" * 55)
        if std_hash and fast_hash and std_hash == fast_hash:
            print("RESULT: EXACT MATCH (Mathematically Sound!)")
        else:
            print("RESULT: MISMATCH! (Data corruption detected)")
        print("="*55)

if __name__ == "__main__":
    main()
