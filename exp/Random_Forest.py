import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import pickle
from Decision_Tree_works import DecisionTree

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



class RandomForest:
    def __init__(self, n_trees=10, criterion="gini", min_samples_leaf=0.05, min_samples_split=0.1, max_features="sqrt", bootstrap=True, random_state=None):
        """
        Initialize the Random Forest model.

        Args:
            n_estimators (int): Number of decision trees in the forest.
            criterion (str): Splitting criterion for each tree ("gini", "entropy", or "f_test").
            min_samples_leaf (float): Minimum fraction of observations in a leaf (pre-pruning).
            min_samples_split (float): Minimum fraction of observations required to search for a split (pre-pruning).
            max_features (str or int): Number of features to consider when looking for the best split. "sqrt" or "log2".
            bootstrap (bool): Whether to use bootstrapped samples for training each tree.
            random_state (int or None): Random seed for reproducibility.
        """
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
        """Generate a bootstrap sample of the data."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def _get_feature_subset(self, n_features):
        """Select a random subset of features."""
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        

        return np.random.choice(n_features, max_features, replace=False)

    def fit(self, X, y):
        """Train the Random Forest model."""
        np.random.seed(self.random_state)
        n_features = X.shape[1]

        for i in range(self.n_trees):
            # Bootstrap sampling
            
            X_sample, y_sample = self._bootstrap_sample(X, y)
            

            # Feature subset
            feature_subset = self._get_feature_subset(n_features)
            self.feature_subsets.append(feature_subset)

            # Train a decision tree on the bootstrap sample and feature subset
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
        """Evaluate the model using sklearn metrics."""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        return accuracy, precision, recall, f1

    def save_model(self, filepath):
        """Save the model to a file."""
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        print(f"Model saved to {filepath}")

    
    def load_model(filepath):
        """Load a model from a file."""
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        print(f"Model loaded from {filepath}")
        return model
    


    def load_data_from_csv(file_path):
        """Load and preprocess data from a CSV file."""
        df = pd.read_csv(file_path)
        
        preprocessor = DataPreprocessor()
        return preprocessor.preprocess(df)

    def test_ann_with_csv(file_path):
        """Test the decision tree model using data from a CSV file."""
        X_train, X_test, y_train, y_test= RandomForest.load_data_from_csv(file_path)

        model = RandomForest(criterion="f_test", min_samples_leaf=0.05, min_samples_split=0.1) # gini works, entropy works
        
        print("Training the model...")
        model.fit(X_train, y_train)

        print("\nEvaluating on test data...")
        accuracy, precision, recall, f1 = model.evaluate(X_test, y_test)
        
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

# Example usage
if __name__ == "__main__":
    csv_file_path =  "C:/Users/cypri/OneDrive/Desktop/TUB/Research Project - Google/code/big_data.csv" 
    RandomForest.test_ann_with_csv(csv_file_path)