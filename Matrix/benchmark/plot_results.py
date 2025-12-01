import json
import matplotlib.pyplot as plt
import os
import sys

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_JSON = os.path.join(RESULTS_DIR, "benchmark_results.json")

def plot_results():
    if not os.path.exists(RESULTS_JSON):
        print(f"Results file not found: {RESULTS_JSON}")
        return

    with open(RESULTS_JSON, 'r') as f:
        results = json.load(f)

    sizes = [r['size'] for r in results]
    
    operations = ['multiplication', 'addition', 'transpose']
    
    for op in operations:
        cpp_times = [r['cpp'][op] for r in results]
        numpy_times = [r['numpy'][op] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, cpp_times, marker='o', label='C++ Implementation')
        plt.plot(sizes, numpy_times, marker='x', label='Numpy Implementation')
        
        plt.title(f'Matrix {op.capitalize()} Benchmark')
        plt.xlabel('Matrix Size (NxN)')
        plt.ylabel('Time (seconds)')
        plt.legend()
        plt.grid(True)
        
        output_path = os.path.join(RESULTS_DIR, f'benchmark_{op}.png')
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    plot_results()
