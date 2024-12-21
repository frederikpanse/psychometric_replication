import numpy as np
from scipy.stats import norm
from scipy.special import softmax
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import pandas as pd
from sklearn.preprocessing import LabelEncoder

class LogisticRegression:
    def __init__(self, link_function="logit"):
        
        if link_function not in ["logit", "probit"]:
            raise ValueError("link_function must be either 'logit' or 'probit'")
        self.link_function = link_function
        self.weights = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _probit_cdf(self, z):
        return norm.cdf(z)   #0.5 * (1 + np.erf(z / np.sqrt(2))) 

    def _predict_proba(self, z):
        if self.link_function == "logit":
            return self._sigmoid(z)
        elif self.link_function == "probit":
            return self._probit_cdf(z)

    def fit(self, X, y, learning_rate=0.1, max_iter=1000, tolerance=1e-6): #We use the same learning rate as in ANN, maybe ask Jeremy about it
        
        X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Initialize weights
        if n_classes > 2:
            # Multi-class classification: One-vs-rest encoding
            y_encoded = np.zeros((n_samples, n_classes))
            for i, label in enumerate(y):
                y_encoded[i, label] = 1
            y = y_encoded
            self.weights = np.zeros((n_features, n_classes))
        else:
            # Binary classification
            self.weights = np.zeros(n_features)

        for iteration in range(max_iter):
            # Compute linear combination
            linear_model = np.dot(X, self.weights)

            # Compute probabilities
            probabilities = self._predict_proba(linear_model)

            # Compute gradient
            if n_classes > 2:
                gradient = np.dot(X.T, (probabilities - y)) / n_samples
            else:
                gradient = np.dot(X.T, (probabilities - y)) / n_samples

            # Update weights
            self.weights -= learning_rate * gradient

        #     # Check for convergence
        #     if np.linalg.norm(gradient) < tolerance:
        #         print(f"Converged after {iteration + 1} iterations.")
        #         break
        # else:
        #     print("Gradient descent did not converge within the maximum iterations.")

    def predict_proba(self, X):
        """
        Predict probabilities for the given input data.

        Args:
            X (numpy.ndarray): Input features (shape: [n_samples, n_features]).

        Returns:
            numpy.ndarray: Predicted probabilities (shape: [n_samples, n_classes]).
        """
        X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
        linear_model = np.dot(X, self.weights)

        if self.weights.ndim > 1 and self.weights.shape[1] > 1:  # Multi-class
            probabilities = softmax(linear_model, axis=1)
        else:  # Binary classification
            probabilities = self._predict_proba(linear_model)
            probabilities = np.column_stack([1 - probabilities, probabilities])
        return probabilities

    def predict(self, X):
        """
        Predict labels for the given input data.

        Args:
            X (numpy.ndarray): Input features (shape: [n_samples, n_features]).

        Returns:
            numpy.ndarray: Predicted labels (shape: [n_samples]).
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        """
        Compute accuracy of the model.

        Args:
            X (numpy.ndarray): Input features (shape: [n_samples, n_features]).
            y (numpy.ndarray): True labels (shape: [n_samples]).

        Returns:
            float: Accuracy of the model.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def evaluate(self, X, y):
        """
        Evaluate the model using sklearn metrics.

        Args:
            X (numpy.ndarray): Input features (shape: [n_samples, n_features]).
            y (numpy.ndarray): True labels (shape: [n_samples]).

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        accuracy =  accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted')
        recall = recall_score(y, predictions, average='weighted')
        f1 = f1_score(y, predictions, average='weighted')
        
        
        return accuracy, precision, recall, f1

    def save_model(self, filepath):
        """
        Save the model to a file.

        Args:
            filepath (str): Path to save the model.
        """
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """
        Load a model from a file.

        Args:
            filepath (str): Path to the model file.

        Returns:
            LogisticRegression: Loaded model instance.
        """
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        print(f"Model loaded from {filepath}")
        return model

# Example usage
# X_train = np.array([[...], [...], ...])
# y_train = np.array([...])
# model = LogisticRegression(link_function="logit")
# model.fit(X_train, y_train, learning_rate=0.1, max_iter=1000, tolerance=1e-6)
# predictions = model.predict(X_test)
# metrics = model.evaluate(X_test, y_test)
# model.save_model("logistic_model.pkl")
# loaded_model = LogisticRegression.load_model("logistic_model.pkl")



    def load_data_from_csv(file_path):
        # Load data from csv file, later update so it get it from github url,  
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1].values  # Features (all columns except the last one)
        y = df.iloc[:, -1].values   # Target (the last column)
        # Scale features
        


    # Encode labels if necessary (already binary in this example)
        le = LabelEncoder()
        y = le.fit_transform(y)


        return X, y

    def test_ann_with_csv(file_path):
    # Load and preprocess the data
        X_data, y_data = LogisticRegression.load_data_from_csv(file_path)
            
            # Define input and output dimensions, hidden layers, activation function, and optimize
            
            # Initialize the model
        model = LogisticRegression(link_function='probit')

            # Train the model
        print("Training the model...")
        model.fit(X_data, y_data)

            # Evaluate the model on test data
        print("\nEvaluating on test data...")
        print(model.evaluate(X_data, y_data))

    

# Test the model with a CSV file (replace with your actual file path)
csv_file_path = "C:/Users/cypri/OneDrive/Desktop/TUB/Research Project - Google/code/first_data.csv"  # Update this path
LogisticRegression.test_ann_with_csv(csv_file_path)
