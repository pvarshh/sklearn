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
| 1000 | 5 | 100.00 | 47.12 | 4.69 |
| 5000 | 5 | 100.00 | 9.97 | 3.50 |
| 10000 | 10 | 100.00 | 14.57 | 3.49 |
| 15000 | 10 | 99.97 | 21.93 | 3.56 |
| 20000 | 10 | 100.00 | 23.00 | 2.78 |
| 25000 | 10 | 100.00 | 11.21 | 2.70 |
| 30000 | 10 | 99.95 | 8.14 | 2.71 |
| 40000 | 10 | 99.99 | 10.36 | 2.55 |
| 50000 | 10 | 99.99 | 9.10 | 2.43 |
| 100000 | 10 | 100.00 | 4.35 | 1.32 |

## Performance Graph

![Benchmark Plot](results/benchmark_plot.png)

## How to Run

```bash
./run_benchmark.sh
```
