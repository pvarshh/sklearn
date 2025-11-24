import random
import argparse
import os

def generate_data(num_samples, num_features, num_classes, output_file):
    """
    Generates synthetic classification data and writes it to a CSV file.
    """
    with open(output_file, 'w') as f:
        # 1. Write Metadata (Header)
        header = [f"feature_{i}" for i in range(num_features)]
        header.append("label")
        f.write(",".join(header) + "\n")
        
        # 2. Generate Data
        for _ in range(num_samples):
            # Generate random integer features between 0 and 100
            features = [str(random.randint(0, 100)) for _ in range(num_features)]
            
            # Generate random label
            label_idx = random.randint(0, num_classes - 1)
            label = f"Class{label_idx}"
            
            # Write row
            row = features + [label]
            f.write(",".join(row) + "\n")

def generate_train_test(num_samples, num_features, num_classes, train_file, test_file, test_ratio=0.2):
    """
    Generates train and test datasets.
    """
    num_test = int(num_samples * test_ratio)
    
    print(f"Generating {num_samples} training samples and {num_test} test samples...")
    
    generate_data(num_samples, num_features, num_classes, train_file)
    generate_data(num_test, num_features, num_classes, test_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for KNN")
    parser.add_argument("--samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--features", type=int, default=5, help="Number of features per sample")
    parser.add_argument("--classes", type=int, default=3, help="Number of classes")
    parser.add_argument("--train_out", type=str, default="train.csv", help="Train output file path")
    parser.add_argument("--test_out", type=str, default="test.csv", help="Test output file path")
    
    args = parser.parse_args()
    
    generate_train_test(args.samples, args.features, args.classes, args.train_out, args.test_out)
