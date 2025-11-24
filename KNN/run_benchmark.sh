#!/bin/bash

# Ensure we are in the KNN directory
cd "$(dirname "$0")"

# Create results directory if it doesn't exist
mkdir -p results

# Run the python benchmark script
python3 benchmark/run_benchmark.py
