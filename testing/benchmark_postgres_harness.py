import subprocess
import re
import argparse
import time
import shutil
import os
from statistics import mean

def run_cmd(cmd, check=True, capture=False):
    """Executes a shell command and optionally captures output."""
    result = subprocess.run(cmd, shell=True, text=True, 
                            capture_output=capture, check=check)
    return result.stdout if capture else None

def parse_pgbench_output(output):
    """Extracts TPS and Latency from pgbench output."""
    tps_match = re.search(r"tps = ([\d\.]+)", output)
    lat_match = re.search(r"latency average = ([\d\.]+)", output)
    
    tps = float(tps_match.group(1)) if tps_match else 0.0
    lat = float(lat_match.group(1)) if lat_match else 0.0
    return tps, lat

def setup_db(bin_dir):
    """Initializes and starts a fresh database, then runs pgbench init."""
    print("  [Setup] Initializing and starting DB...")
    if os.path.exists("./test_db"):
        shutil.rmtree("./test_db")
        
    run_cmd(f"{bin_dir}/initdb -D ./test_db", capture=True)
    run_cmd(f"{bin_dir}/pg_ctl -D ./test_db -l logfile -w start", capture=True)
    run_cmd(f"{bin_dir}/createdb test_perf", capture=True)
    run_cmd(f"{bin_dir}/pgbench -i -s 10 test_perf", capture=True)

def teardown_db(bin_dir):
    """Stops and destroys the database."""
    print("  [Teardown] Stopping and cleaning DB...")
    run_cmd(f"{bin_dir}/pg_ctl -D ./test_db stop", check=False, capture=True)
    if os.path.exists("./test_db"):
        shutil.rmtree("./test_db")

def main():
    parser = argparse.ArgumentParser(description="PostgreSQL Benchmark Harness")
    parser.add_argument("-r", "--runs", type=int, default=3, help="Number of benchmark iterations to run")
    parser.add_argument("-b", "--bin-dir", type=str, default="./bin", help="Path to postgres bin directory")
    args = parser.parse_args()

    metrics = {
        'write_tps': [], 'write_lat': [],
        'read_tps': [], 'read_lat': []
    }

    print(f"=== Starting Benchmark Harness ({args.runs} Runs) ===")

    for i in range(args.runs):
        print(f"\n--- Run {i+1} of {args.runs} ---")
        
        setup_db(args.bin_dir)
        print("  Running Write Test (TPC-B)...")
        write_out = run_cmd(f"{args.bin_dir}/pgbench -c 10 -j 2 -T 60 test_perf", capture=True)
        w_tps, w_lat = parse_pgbench_output(write_out)
        metrics['write_tps'].append(w_tps)
        metrics['write_lat'].append(w_lat)
        print(f"    Result -> TPS: {w_tps:.2f}, Latency: {w_lat:.3f} ms")
        teardown_db(args.bin_dir)

        setup_db(args.bin_dir)
        print("  Running Read Test (Select Only)...")
        read_out = run_cmd(f"{args.bin_dir}/pgbench -S -c 20 -j 4 -T 60 test_perf", capture=True)
        r_tps, r_lat = parse_pgbench_output(read_out)
        metrics['read_tps'].append(r_tps)
        metrics['read_lat'].append(r_lat)
        print(f"    Result -> TPS: {r_tps:.2f}, Latency: {r_lat:.3f} ms")
        teardown_db(args.bin_dir)

    print("\n=========================================================")
    print("                 FINAL AGGREGATE RESULTS                 ")
    print("=========================================================")
    
    def print_stats(name, data, unit):
        avg_val = mean(data)
        min_val = min(data)
        max_val = max(data)
        print(f"{name:15} | Avg: {avg_val:9.2f} | Min: {min_val:9.2f} | Max: {max_val:9.2f} {unit}")

    print_stats("Write TPS", metrics['write_tps'], "TPS")
    print_stats("Read TPS", metrics['read_tps'], "TPS")
    print("-" * 57)
    print_stats("Write Latency", metrics['write_lat'], "ms")
    print_stats("Read Latency", metrics['read_lat'], "ms")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
