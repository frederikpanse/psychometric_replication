import numpy as np
from scipy.stats import f_oneway
from collections import Counter
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self, test_size=0.3, random_state=3103, stratify=True):
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def preprocess(self, data):
        # Separate features and target
        X = data.filter(like='X')  
        y = data['y_cat'] 
        
        # Apply label encoding to y
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=self.test_size, random_state=self.random_state, stratify=y if self.stratify else None
        )
        ### Ask Jeremy if about standardizing and stratification
        # Standardize the features
        X_train_scaled = self.scaler.fit_transform(X_train)  
        X_test_scaled = self.scaler.transform(X_test)  
        
        return X_train_scaled, X_test_scaled, y_train, y_test




class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, criterion="gini", min_samples_leaf=0.05, min_samples_split=0.1):
        """
        Initialize the Decision Tree model.

        Args:
            criterion (str): Splitting criterion ("gini", "entropy", or "f_test").
            min_samples_leaf (float): Minimum fraction of observations in a leaf (pre-pruning).
            min_samples_split (float): Minimum fraction of observations required to search for a split (pre-pruning).
        """
        if criterion not in ["gini", "entropy", "f_test"]:
            raise ValueError("criterion must be one of 'gini', 'entropy', or 'f_test'")
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        """Calculate Gini impurity."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _entropy(self, y):
        """Calculate entropy."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum(p * np.log2(p) for p in probabilities if p > 0)

    def _f_test(self, X, y):
        """Perform F-test for splitting."""
        unique_classes = np.unique(y)
        groups = [X[y == c] for c in unique_classes]
        if len(groups) < 2:  # F-test requires at least two groups
            return 0
        f_stat, _ = f_oneway(*groups)
        return f_stat

    def _split_criterion(self, X_left, X_right, y_left, y_right):
        n_left = len(y_left)
        n_right = len(y_right)
        total = n_left + n_right

        if n_left == 0 or n_right == 0:
            return float('inf')

        if self.criterion == "gini":
            score_left = self._gini(y_left)
            score_right = self._gini(y_right)
            return (n_left / total) * score_left + (n_right / total) * score_right
        elif self.criterion == "entropy":
            score_left = self._entropy(y_left)
            score_right = self._entropy(y_right)
            return (n_left / total) * score_left + (n_right / total) * score_right
        elif self.criterion == "f_test":
            combined_X = np.vstack([X_left, X_right])
            combined_y = np.hstack([y_left, y_right])
            score = self._f_test(combined_X, combined_y)
            return -score  # Negative F-statistic to minimize

    def best_split(self, X, y):
        num_features = X.shape[1]
        optimal_feature = None
        optimal_threshold = None
        best_score = float('inf')  # Initialize with infinity

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if np.any(left_indices) and np.any(right_indices):
                    X_left, X_right = X[left_indices], X[right_indices]
                    y_left, y_right = y[left_indices], y[right_indices]

                    score = self._split_criterion(X_left, X_right, y_left, y_right)

                # Update the best split
                    if np.isscalar(score) and score < best_score:
                        best_score = score
                        optimal_feature = feature
                        optimal_threshold = threshold

        return optimal_feature, optimal_threshold


    def build_tree(self, X, y):
    
    # Stop splitting if the number of samples is less than min_samples_split
        if len(y) < int(self.min_samples_split * len(y)):
            return Node(value=np.bincount(y).argmax())
    
    # Stop splitting if all labels are the same
        if len(np.unique(y)) == 1:
            return Node(value=np.bincount(y).argmax())

        optimal_feature, optimal_threshold = self.best_split(X, y)

        if optimal_feature is None:
            return Node(value=np.bincount(y).argmax())

        left_indices = X[:, optimal_feature] <= optimal_threshold
        right_indices = X[:, optimal_feature] > optimal_threshold

    # Stop splitting if resulting nodes would have fewer samples than min_samples_leaf
        if sum(left_indices) < int(self.min_samples_leaf * len(y)) or sum(right_indices) < int(self.min_samples_leaf * len(y)):
            return Node(value=np.bincount(y).argmax())

        left_subtree = self.build_tree(X[left_indices], y[left_indices])
        right_subtree = self.build_tree(X[right_indices], y[right_indices])

        return Node(feature=optimal_feature, threshold=optimal_threshold, left=left_subtree, right=right_subtree)


    def fit(self, X, y):
        """Fit the decision tree model."""
        self.root = self.build_tree(X, y)

    def predict_sample(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_sample(node.left, x)
        else:
            return self.predict_sample(node.right, x)

    def predict(self, X):
        """Predict the labels for the input data."""
        return np.array([self.predict_sample(self.root, x) for x in X])

    def evaluate(self, X, y):
        """Evaluate the model using sklearn metrics."""
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        return accuracy, precision, recall, f1

    def save_model(self, filepath):
        """Save the model to a file."""
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """Load a model from a file."""
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        print(f"Model loaded from {filepath}")
        return model

    @staticmethod
    def load_data_from_csv(file_path):
        """Load and preprocess data from a CSV file."""
        df = pd.read_csv(file_path)
        
        preprocessor = DataPreprocessor()
        return preprocessor.preprocess(df)

    @staticmethod
    def test_ann_with_csv(file_path):
        """Test the decision tree model using data from a CSV file."""
        X_train_data, X_test_data, y_train_data, y_test_data = DecisionTree.load_data_from_csv(file_path)

        model = DecisionTree(criterion="f_test", min_samples_leaf=0.05, min_samples_split=0.1) # gini works, entropy works
        
        print("Training the model...")
        model.fit(X_train_data, y_train_data)

        print("\nEvaluating on test data...")
        accuracy, precision, recall, f1 = model.evaluate(X_test_data, y_test_data)
        
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

# Example usage
csv_file_path =  "C:/Users/cypri/OneDrive/Desktop/TUB/Research Project - Google/code/big_data.csv" 
DecisionTree.test_ann_with_csv(csv_file_path)
