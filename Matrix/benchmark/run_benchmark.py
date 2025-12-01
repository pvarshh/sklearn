import sys
import os
import time
import json
import subprocess
import numpy as np
import pandas as pd

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPP_EXECUTABLE = os.path.join(BASE_DIR, "matrix_benchmark")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
MATRIX_A_FILE = os.path.join(DATA_DIR, "matrix_A.csv")
MATRIX_B_FILE = os.path.join(DATA_DIR, "matrix_B.csv")
RESULTS_JSON = os.path.join(RESULTS_DIR, "benchmark_results.json")

def compile_cpp():
    print("Compiling C++ implementation...")
    cmd = [
        "g++", "-O3", "-std=c++11", "-march=native", "-pthread", "-ffast-math", "-funroll-loops",
        "-o", CPP_EXECUTABLE,
        os.path.join(BASE_DIR, "benchmark/benchmark_main.cpp"),
        os.path.join(BASE_DIR, "cpp_implementation/Matrix.cpp"),
        "-I", os.path.join(BASE_DIR, "cpp_implementation")
    ]
    subprocess.check_call(cmd)
    print("Compilation successful.")

def generate_data(size):
    print(f"Generating data for size {size}...")
    cmd = [
        "python3",
        os.path.join(BASE_DIR, "data/generate_data.py"),
        "--size", str(size),
        "--output_dir", DATA_DIR
    ]
    subprocess.check_call(cmd)

def run_cpp_benchmark():
    print("Running C++ benchmark...")
    cmd = [CPP_EXECUTABLE, MATRIX_A_FILE, MATRIX_B_FILE]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    output = result.stdout
    print(output) # Print C++ output for debugging
    times = {}
    for line in output.splitlines():
        if "Multiplication Time:" in line:
            times["multiplication"] = float(line.split(":")[1].strip().split()[0])
        elif "Addition Time:" in line:
            times["addition"] = float(line.split(":")[1].strip().split()[0])
        elif "Transpose Time:" in line:
            times["transpose"] = float(line.split(":")[1].strip().split()[0])
    return times

def run_numpy_benchmark(size):
    print("Running Numpy benchmark...")
    # We can just run the numpy operations here directly since we have numpy
    # But to be fair, we should load the same CSVs? 
    # Loading CSVs in python is slow and might dominate the benchmark if we include it.
    # The C++ benchmark includes reading CSV? No, let's check benchmark_main.cpp again.
    # Yes, it reads CSV.
    # To be fair, we should measure only the operation time.
    
    # Load data
    df_A = pd.read_csv(MATRIX_A_FILE, header=None)
    df_B = pd.read_csv(MATRIX_B_FILE, header=None)
    A = df_A.values
    B = df_B.values
    
    times = {}
    
    start = time.time()
    np.dot(A, B)
    times["multiplication"] = time.time() - start
    
    start = time.time()
    np.add(A, B)
    times["addition"] = time.time() - start
    
    start = time.time()
    np.transpose(A)
    times["transpose"] = time.time() - start
    
    return times

def run_benchmark():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    compile_cpp()
    
    sizes = [100, 200, 500, 1000] # Adjust sizes as needed
    results = []
    
    for size in sizes:
        print(f"\n--- Benchmarking size {size} ---")
        generate_data(size)
        
        cpp_times = run_cpp_benchmark()
        numpy_times = run_numpy_benchmark(size)
        
        results.append({
            "size": size,
            "cpp": cpp_times,
            "numpy": numpy_times
        })
        
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nBenchmark complete. Results saved to {RESULTS_JSON}")

if __name__ == "__main__":
    run_benchmark()
