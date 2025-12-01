#!/bin/bash

# Exit on error
set -e

echo "Running Matrix Benchmark..."

# Run the benchmark script
python3 benchmark/run_benchmark.py

# Plot the results
python3 benchmark/plot_results.py

echo "Done!"
