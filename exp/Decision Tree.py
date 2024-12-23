import numpy as np
from scipy.stats import f_oneway
from collections import Counter
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, criterion="gini", min_samples_leaf=0.05, min_samples_split=0.1):
        
        if criterion not in ["gini", "entropy", "f_test"]:
            raise ValueError("criterion must be one of 'gini', 'entropy', or 'f_test'")
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        # Implement Gini Impurity formula
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _entropy(self, y):
        # Implement Entropy formula
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
        best_score = float('inf')  # Initialize with infinity, ensures that any real number will be smaller than the initial value. Standar approach when searching for minimum score

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if np.any(left_indices) and np.any(right_indices):
                    X_left, X_right = X[left_indices], X[right_indices]
                    y_left, y_right = y[left_indices], y[right_indices]

                    score = self._split_criterion(X_left, X_right, y_left, y_right)

                # Update the best split, np.isscalar ensures that there is no error 
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

    # Stop splitting if resulting nodes have fewer samples than min_samples_leaf
        if sum(left_indices) < int(self.min_samples_leaf * len(y)) or sum(right_indices) < int(self.min_samples_leaf * len(y)):
            return Node(value=np.bincount(y).argmax())

        left_subtree = self.build_tree(X[left_indices], y[left_indices])
        right_subtree = self.build_tree(X[right_indices], y[right_indices])

        return Node(feature=optimal_feature, threshold=optimal_threshold, left=left_subtree, right=right_subtree)


    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_sample(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_sample(node.left, x)
        else:
            return self.predict_sample(node.right, x)

    def predict(self, X):
        return np.array([self.predict_sample(self.root, x) for x in X])

    def evaluate(self, X_test, y_test):
        
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        return accuracy, precision, recall, f1

    def save_model(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        print(f"Model saved to {filepath}")

    
    def load_model(filepath):
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        print(f"Model loaded from {filepath}")
        return model


    

'''
You can use the above code using sample code:
model = DecisionTree(criterion="gini", min_samples_leaf=0.05, min_samples_split=0.1)
model.fit(X_train, y_train)
accuracy, precision, recall, f1 = model.evaluate(X_test, y_test)

'''
