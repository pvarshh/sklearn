# K-Nearest Neighbors (KNN) Implementation

This directory contains a C++ implementation of the K-Nearest Neighbors algorithm optimized using a KD-Tree.

## Assumptions

Suppose there are `n` features needed for a classification, and `k` data points in the data.

This means each `data point` (represented by a vector) will have `size = n + 1`

<b> Data Formating </b>
<ol>
<li> 1st row <b>always</b> refers to the metadata
<li> Data follows [feature<sup>k<sub>i</sub></sup><sub>0</sub>, feature<sup>k<sub>i</sub></sup><sub>1</sub>, ... feature<sup>k<sub>i</sub></sup><sub>n</sub>, label<sub>i</sub>] so processing is easy and uniform
</ol>

## Implementation Details

### KD-Tree Optimization
Instead of a brute-force $O(N)$ search for every query, this implementation uses a **KD-Tree (k-dimensional tree)** to organize the training data. A KD-Tree is a binary tree that splits points at each level by a specific dimension (axis), cycling through dimensions (x, y, z, etc.).

- **Build Process**: The tree is built recursively. At each step, the data is sorted along the current axis, and the median point becomes the node. This ensures a balanced tree.
- **Search Process**: The search is recursive. It traverses down to the leaf node closest to the query point. As it unwinds, it checks if it needs to explore the other side of the splitting plane (pruning).

### Key Components

#### `KDNode` Struct
Represents a node in the KD-Tree.
- `point`: The feature vector of the data point.
- `label`: The class label associated with the point.
- `left`, `right`: Pointers to child nodes.
- `axis`: The dimension used to split data at this node.

#### `KNN` Class
- **`fit(X, y)`**: Builds the KD-Tree from in-memory vectors.
- **`fit(filepath)`**: Loads data from a file (CSV or space-separated) and builds the KD-Tree.
  - Skips the first row (metadata).
  - Expects the last column to be the label.
  - Uses `std::nth_element` to find the median in $O(N)$ time, making the total build time $O(N \log N)$.
- **`classify(point)`**: Predicts the label for a single query point.
  - Uses a priority queue (max-heap) to keep track of the $k$ nearest neighbors found so far.
  - Prunes search branches that cannot possibly contain a closer point than the current $k$-th nearest neighbor.

### Complexity Analysis

| Operation | Brute Force | KD-Tree (Average) | KD-Tree (Worst Case) |
|-----------|-------------|-------------------|----------------------|
| **Training (Fit)** | $O(N \cdot D)$ | $O(N \cdot D + N \log N)$ | $O(N \cdot D + N \log N)$ |
| **Prediction** | $O(N \cdot D)$ | $O(D \cdot \log N)$ | $O(N \cdot D)$ |
| **Space** | $O(N \cdot D)$ | $O(N \cdot D)$ | $O(N \cdot D)$ |

Where $N$ is the number of samples and $D$ is the number of features.
*Note: Brute Force training complexity is $O(N \cdot D)$ due to data storage/copying. If data is merely referenced, it is $O(1)$.*

### Usage

```cpp
#include "KNN.h"
#include <iostream>

int main() {
    // 1. Initialize KNN with k=3
    KNN knn(3);

    // 2. Train the model
    try {
        knn.fit("data.csv"); 
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << std::endl;
        return 1;
    }

    // 3. Classify a new point
    std::vector<int> new_point = {2, 3};
    std::string prediction = knn.classify(new_point);
    
    std::cout << "Predicted label: " << prediction << std::endl;
    
    return 0;
}
```
