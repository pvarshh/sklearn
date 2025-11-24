# KNN Implementation Benchmark

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
| 1000 | 5 | 99.50 | 46.51 | 4.59 |
| 5000 | 5 | 100.00 | 10.19 | 4.32 |
| 10000 | 10 | 100.00 | 15.62 | 3.65 |
| 15000 | 10 | 100.00 | 12.86 | 3.33 |
| 20000 | 10 | 100.00 | 9.46 | 3.29 |
| 25000 | 10 | 99.98 | 9.62 | 3.04 |
| 30000 | 10 | 99.95 | 9.91 | 3.12 |
| 40000 | 10 | 99.98 | 7.32 | 3.03 |
| 50000 | 10 | 99.98 | 3.87 | 2.49 |
| 100000 | 10 | 99.98 | 5.77 | 1.38 |

## Performance Graph

![Benchmark Plot](results/benchmark_plot.png)

## How to Run

```bash
./run_benchmark.sh
```
