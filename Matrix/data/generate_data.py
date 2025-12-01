import numpy as np
import pandas as pd
import os
import argparse

def generate_matrix_data(rows, cols, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Generating {rows}x{cols} matrices...")
    matrix_A = np.random.rand(rows, cols)
    matrix_B = np.random.rand(cols, rows) # For multiplication A * B, B needs to be cols x rows (resulting in rows x rows) or similar. 
                                          # Let's stick to square matrices for simplicity in this benchmark or make B compatible.
                                          # If A is rows x cols, B should be cols x something. Let's make B cols x rows so result is rows x rows.
    
    # Actually, for addition A+B, they must be same size.
    # For multiplication A*B, cols(A) == rows(B).
    # To support both with same pair of matrices, they must be square.
    
    matrix_B_square = np.random.rand(rows, cols)

    pd.DataFrame(matrix_A).to_csv(os.path.join(output_dir, 'matrix_A.csv'), index=False, header=False)
    pd.DataFrame(matrix_B_square).to_csv(os.path.join(output_dir, 'matrix_B.csv'), index=False, header=False)
    
    print(f"Saved matrix_A.csv and matrix_B.csv to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random matrices.')
    parser.add_argument('--size', type=int, default=100, help='Size of the square matrices')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    generate_matrix_data(args.size, args.size, args.output_dir)
