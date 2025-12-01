import numpy as np
import time

def run_matrix_operations(size):
    """
    Performs matrix operations using NumPy.
    """
    print(f"Generating {size}x{size} matrices...")
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    print("Performing Matrix Multiplication...")
    start_time = time.time()
    C = np.dot(A, B)
    end_time = time.time()
    print(f"Multiplication took: {end_time - start_time:.6f} seconds")

    print("Performing Matrix Addition...")
    start_time = time.time()
    D = np.add(A, B)
    end_time = time.time()
    print(f"Addition took: {end_time - start_time:.6f} seconds")

    print("Performing Matrix Transpose...")
    start_time = time.time()
    E = np.transpose(A)
    end_time = time.time()
    print(f"Transpose took: {end_time - start_time:.6f} seconds")

    return C, D, E

if __name__ == "__main__":
    run_matrix_operations(1000)
