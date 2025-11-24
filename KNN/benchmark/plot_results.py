import matplotlib.pyplot as plt
import json
import os
import sys

def plot_and_update_readme():
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_dir, "results", "benchmark_results.json")
    plot_path = os.path.join(base_dir, "results", "benchmark_plot.png")
    readme_path = os.path.join(base_dir, "README.md")

    if not os.path.exists(results_path):
        print(f"Results file not found at {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    # Extract data
    n_samples = [r['config']['n_samples'] for r in results]
    
    # Calculate Speedups
    # Speedup = Sklearn Time / C++ Time
    single_speedups = []
    batch_speedups = []
    
    for r in results:
        # Single
        t_sk_single = r['sklearn']['single_pred']
        t_cpp_single = r['cpp']['single_pred']
        if t_cpp_single > 0:
            single_speedups.append(t_sk_single / t_cpp_single)
        else:
            single_speedups.append(0) # Should not happen
            
        # Batch
        t_sk_batch = r['sklearn']['batch_pred']
        t_cpp_batch = r['cpp']['batch_pred']
        if t_cpp_batch > 0:
            batch_speedups.append(t_sk_batch / t_cpp_batch)
        else:
            batch_speedups.append(0)

    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot Speedups
    plt.plot(n_samples, single_speedups, marker='o', label='Single Prediction Speedup', linewidth=2, color='green')
    plt.plot(n_samples, batch_speedups, marker='s', label='Batch Prediction Speedup', linewidth=2, color='purple')

    # Add baseline at 1.0 (Equal Performance)
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3, label='Baseline (Equal Speed)')
    plt.text(min(n_samples), 1.1, 'C++ Faster', color='green', fontsize=10, fontweight='bold')
    plt.text(min(n_samples), 0.9, 'Sklearn Faster', color='red', fontsize=10, fontweight='bold')

    plt.xlabel('Number of Training Samples (N)', fontsize=12)
    plt.ylabel('Speedup Factor (Sklearn Time / C++ Time)', fontsize=12)
    plt.title('C++ Speedup over Scikit-learn', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    plt.xscale('log')
    
    # Y-axis log scale might be useful if single speedup is huge
    # But linear might be easier to read for "2x, 3x". 
    # Given Single is ~40x and Batch is ~4x, log scale is better to see both.
    plt.yscale('log')
    
    # Custom Y-ticks for log scale to make it readable
    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    
    plt.savefig(plot_path)
    print(f"Graph saved as {plot_path}")

    # Update README
    update_readme(readme_path, results)

def update_readme(readme_path, results):
    markdown_content = """# KNN Implementation Benchmark

This project compares a custom C++ K-Nearest Neighbors implementation (using KD-Tree and Multi-threading) against Scikit-learn's implementation (configured to use Brute Force search for high-dimensional efficiency).

## Implementations

1.  **C++ Implementation**:
    *   **Algorithm**: KD-Tree (k-dimensional tree) for spatial indexing.
    *   **Parallelism**: Uses `std::async` and `std::future` for multi-threaded batch classification.
    *   **Optimization**: `std::nth_element` for efficient tree construction.

2.  **Scikit-learn Implementation**:
    *   **Algorithm**: `brute` (Brute Force).
    *   **Metric**: Euclidean distance.
    *   **Note**: Brute force is often preferred in high dimensions due to the "curse of dimensionality" making tree structures less effective, but our optimized C++ KD-Tree still demonstrates significant speedups.

## Benchmark Results

The following table shows the performance comparison.
*   **Single Speedup**: Speedup factor for predicting a single query point (Latency).
*   **Batch Speedup**: Speedup factor for predicting a large batch of query points (Throughput).

| N (Samples) | k | Agreement (%) | Single Speedup (x) | Batch Speedup (x) |
|:---|:---|:---|:---|:---|
"""
    
    for res in results:
        N = res['config']['n_samples']
        k = res['config']['k']
        agreement = res['agreement']
        
        sk_single = res['sklearn']['single_pred']
        cpp_single = res['cpp']['single_pred']
        single_speedup = sk_single / cpp_single if cpp_single > 0 else 0
        
        sk_batch = res['sklearn']['batch_pred']
        cpp_batch = res['cpp']['batch_pred']
        batch_speedup = sk_batch / cpp_batch if cpp_batch > 0 else 0
        
        markdown_content += f"| {N} | {k} | {agreement:.2f} | {single_speedup:.2f} | {batch_speedup:.2f} |\n"

    markdown_content += """
## Performance Graph

![Benchmark Plot](results/benchmark_plot.png)

## How to Run

```bash
./run_benchmark.sh
```
"""
    
    with open(readme_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"README updated at {readme_path}")

if __name__ == "__main__":
    plot_and_update_readme()
