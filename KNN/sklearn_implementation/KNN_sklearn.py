import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

class KNN:
    def __init__(self, k=3):
        """
        Initialize the KNN classifier.
        
        Args:
            k (int): Number of neighbors to use.
        """
        self.k = k
        # Using 'brute' algorithm with 'euclidean' metric as it is often faster for high-dimensional data
        # due to vectorization, compared to KD-Tree which suffers from curse of dimensionality.
        self.model = KNeighborsClassifier(n_neighbors=k, algorithm='brute', metric='euclidean')
        self.is_fitted = False

    def fit(self, X_or_path, y=None):
        """
        Fit the model using X as training data and y as target values, 
        or load data from a file if X_or_path is a string.

        Args:
            X_or_path (array-like or str): Training data or path to CSV file.
            y (array-like, optional): Target values. Required if X_or_path is array-like.
        """
        if isinstance(X_or_path, str):
            self._fit_from_file(X_or_path)
        else:
            if y is None:
                raise ValueError("y must be provided if X is not a filepath")
            self._fit_from_memory(X_or_path, y)
        
        self.is_fitted = True

    def _fit_from_file(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            # Try reading as CSV first
            df = pd.read_csv(filepath)
            
            # Fallback: try whitespace delimiter if CSV parsing failed to split columns
            if len(df.columns) <= 1:
                 df = pd.read_csv(filepath, delim_whitespace=True)

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            self.model.fit(X, y)
            
        except Exception as e:
            raise ValueError(f"Error loading or parsing file: {e}")

    def _fit_from_memory(self, X, y):
        self.model.fit(X, y)

    def classify(self, points):
        """
        Predict the class labels for the provided data.

        Args:
            points (array-like): Query point (1D) or points (2D).

        Returns:
            str or list: Predicted label(s).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        
        points = np.array(points)
        
        # Handle single point (1D array)
        if points.ndim == 1:
            # Reshape to (1, n_features)
            prediction = self.model.predict(points.reshape(1, -1))
            return prediction[0]
        
        # Handle multiple points (2D array)
        else:
            return self.model.predict(points).tolist()

if __name__ == "__main__":
    # Simple test
    knn = KNN(k=3)
    X_train = [[1, 2], [3, 4], [5, 6]]
    y_train = ["A", "A", "B"]
    knn.fit(X_train, y_train)
    print(f"Prediction for [2, 3]: {knn.classify([2, 3])}")
