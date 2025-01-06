import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import pickle
from Decision_Tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=10, criterion="gini", min_samples_leaf=0.05, min_samples_split=0.1, max_features="sqrt", bootstrap=True, random_state=None):
        
        self.n_trees = n_trees
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_subsets = []

    def _bootstrap_sample(self, X, y):
        # Generate a bootstrap sample of the data
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _get_feature_subset(self, n_features):
        # Select a random subset of features used to train trees
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        

        return np.random.choice(n_features, max_features, replace=False)

    def fit(self, X, y):
        
        np.random.seed(self.random_state)
        n_features = X.shape[1]

        for i in range(self.n_trees):
            # Bootstrap sampling
            
            X_sample, y_sample = self._bootstrap_sample(X, y)
            

            # Feature subset
            feature_subset = self._get_feature_subset(n_features)
            self.feature_subsets.append(feature_subset)

            # Train a decision tree using the bootstrap sample and feature subset
            tree = DecisionTree(criterion=self.criterion, min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split)
            tree.fit(X_sample[:, feature_subset], y_sample)
            self.trees.append(tree)

    def predict(self, X):
        return np.array([self._aggregate_preds(X[i]) for i in range(X.shape[0])])

    def _aggregate_preds(self, x):
        preds = [tree.predict(np.array([x[self.feature_subsets[i]]]))[0] for i, tree in enumerate(self.trees)]
        classes, counts = np.unique(preds, return_counts=True)
        return classes[np.argmax(counts)]
            
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
Yoou can use above code with sample code:
model = RandomForest(criterion="gini", min_samples_leaf=0.05, min_samples_split=0.1)
model.fit(X_train, y_train)
accuracy, precision, recall, f1 = model.evaluate(X_test, y_test)
'''
