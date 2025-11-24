import sys
import os
import time
import json
import subprocess
import pandas as pd
import numpy as np

# Add parent directory to path to import sklearn implementation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn_implementation.KNN_sklearn import KNN as SklearnKNN

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # KNN directory
CPP_EXECUTABLE = os.path.join(BASE_DIR, "knn_benchmark")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
RESULTS_JSON = os.path.join(RESULTS_DIR, "benchmark_results.json")

def compile_cpp():
    print("Compiling C++ implementation...")
    cmd = [
        "g++", "-O3", "-std=c++11",
        "-o", CPP_EXECUTABLE,
        os.path.join(BASE_DIR, "benchmark/benchmark_main.cpp"),
        os.path.join(BASE_DIR, "cpp_implementation/KNN.cpp"),
        "-I", os.path.join(BASE_DIR, "cpp_implementation")
    ]
    subprocess.check_call(cmd)
    print("Compilation successful.")

def run_benchmark():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    compile_cpp()

    configs = [
        {"samples": 1000, "features": 10, "k": 5},
        {"samples": 5000, "features": 20, "k": 5},
        {"samples": 10000, "features": 50, "k": 10},
        {"samples": 15000, "features": 50, "k": 10},
        {"samples": 20000, "features": 50, "k": 10},
        {"samples": 25000, "features": 50, "k": 10},
        {"samples": 30000, "features": 50, "k": 10},
        {"samples": 40000, "features": 50, "k": 10},
        {"samples": 50000, "features": 50, "k": 10},
        {"samples": 100000, "features": 100, "k": 10},
    ]

    results_summary = []

    for config in configs:
        N = config["samples"]
        D = config["features"]
        k = config["k"]
        
        print(f"\nConfiguration: Samples={N}, Features={D}, k={k}")
        print("-" * 40)

        # 1. Generate Data
        subprocess.check_call([
            "python3", os.path.join(BASE_DIR, "data/generate_data.py"),
            "--samples", str(N),
            "--features", str(D),
            "--classes", "5",
            "--train_out", TRAIN_FILE,
            "--test_out", TEST_FILE
        ])

        # Load test labels for agreement check
        test_df = pd.read_csv(TEST_FILE)
        X_test = test_df.iloc[:, :-1].values
        y_test_true = test_df.iloc[:, -1].values

        # 2. Run Scikit-learn
        print("Running scikit-learn...")
        sk_knn = SklearnKNN(k=k)
        
        start_time = time.time()
        sk_knn.fit(TRAIN_FILE)
        sk_train_time = (time.time() - start_time) * 1000 # ms

        # Measure Single Prediction (Average of first 100)
        num_single = min(len(X_test), 100)
        start_time = time.time()
        for i in range(num_single):
            sk_knn.classify(X_test[i])
        sk_single_time = ((time.time() - start_time) * 1000) / num_single

        start_time = time.time()
        sk_batch_preds = sk_knn.classify(X_test)
        sk_batch_time = (time.time() - start_time) * 1000 # ms

        # 3. Run C++
        print("Running C++...")
        try:
            proc = subprocess.Popen(
                [CPP_EXECUTABLE, TRAIN_FILE, TEST_FILE, str(k)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = proc.communicate()
            
            if proc.returncode != 0:
                print(f"C++ Error: {stderr}")
                continue

            lines = stdout.strip().split('\n')
            cpp_train_time = float(lines[0])
            cpp_single_time = float(lines[1])
            cpp_batch_time = float(lines[2])
            cpp_preds = [line.strip() for line in lines[3:] if line.strip()]

        except Exception as e:
            print(f"Failed to run C++: {e}")
            continue

        # 4. Compare
        agreement_count = sum(1 for s, c in zip(sk_batch_preds, cpp_preds) if s == c)
        agreement_pct = (agreement_count / len(sk_batch_preds)) * 100

        # 5. Print Results
        print(f"\nResults for N={N}:")
        print(f"{'Metric':<20} | {'sklearn':<15} | {'C++':<15} | {'Speedup (C++/sklearn)':<20}")
        print("-" * 80)
        
        train_speedup = sk_train_time / cpp_train_time if cpp_train_time > 0 else 0
        single_speedup = sk_single_time / cpp_single_time if cpp_single_time > 0 else 0
        batch_speedup = sk_batch_time / cpp_batch_time if cpp_batch_time > 0 else 0

        print(f"{'Train Time (ms)':<20} | {sk_train_time:<15.4f} | {cpp_train_time:<15.4f} | {train_speedup:<20.2f}")
        print(f"{'Single Pred (ms)':<20} | {sk_single_time:<15.4f} | {cpp_single_time:<15.4f} | {single_speedup:<20.2f}")
        print(f"{'Batch Pred (ms)':<20} | {sk_batch_time:<15.4f} | {cpp_batch_time:<15.4f} | {batch_speedup:<20.2f}")
        print(f"{'Agreement (%)':<20} | {agreement_pct:<15.4f} | {'-':<15} | {'-':<20}")

        results_summary.append({
            "config": {
                "n_samples": N,
                "n_features": D,
                "k": k
            },
            "sklearn": {
                "train": sk_train_time,
                "single_pred": sk_single_time,
                "batch_pred": sk_batch_time
            },
            "cpp": {
                "train": cpp_train_time,
                "single_pred": cpp_single_time,
                "batch_pred": cpp_batch_time
            },
            "agreement": agreement_pct
        })

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"{'N':<10} | {'k':<5} | {'Agreement':<10} | {'Single Speedup':<15} | {'Batch Speedup':<15}")
    print("-" * 80)
    for res in results_summary:
        N = res['config']['n_samples']
        k = res['config']['k']
        agreement = res['agreement']
        
        sk_single = res['sklearn']['single_pred']
        cpp_single = res['cpp']['single_pred']
        single_speedup = sk_single / cpp_single if cpp_single > 0 else 0
        
        sk_batch = res['sklearn']['batch_pred']
        cpp_batch = res['cpp']['batch_pred']
        batch_speedup = sk_batch / cpp_batch if cpp_batch > 0 else 0
        
        print(f"{N:<10} | {k:<5} | {agreement:<10.2f} | {single_speedup:<15.2f} | {batch_speedup:<15.2f}")

    with open(RESULTS_JSON, 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\nResults saved to {RESULTS_JSON}")

    # Generate Plot and Update README
    print("\nGenerating plot and updating README...")
    try:
        subprocess.check_call(["python3", os.path.join(BASE_DIR, "benchmark/plot_results.py")])
    except Exception as e:
        print(f"Failed to generate plot or update README: {e}")
    
    # Cleanup
    print("\n--- Cleanup ---")
    if os.path.exists(TRAIN_FILE): os.remove(TRAIN_FILE)
    if os.path.exists(TEST_FILE): os.remove(TEST_FILE)
    if os.path.exists(CPP_EXECUTABLE): os.remove(CPP_EXECUTABLE)

if __name__ == "__main__":
    run_benchmark()
