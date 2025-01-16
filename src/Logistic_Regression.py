import numpy as np
from scipy.stats import norm
from scipy.special import softmax
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report



class LogisticRegression:
    def __init__(self, link_function="logit"):
        
        if link_function not in ["logit", "probit"]:
            raise ValueError("link_function must be either 'logit' or 'probit'")
        self.link_function = link_function
        self.weights = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _probit_cdf(self, z):
        return norm.cdf(z)   

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
        
        X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term
        linear_model = np.dot(X, self.weights)

        if self.weights.ndim > 1 and self.weights.shape[1] > 1:  # Multi-class
            probabilities = softmax(linear_model, axis=1)
        else:  # Binary classification
            probabilities = self._predict_proba(linear_model)
            probabilities = np.column_stack([1 - probabilities, probabilities])
        return probabilities

    def predict(self, X):
        
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X, y):
        
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def evaluate(self, X, y):
        
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        accuracy =  accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted')
        recall = recall_score(y, predictions, average='weighted')
        f1 = f1_score(y, predictions, average='weighted')
        
        
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
You can use it for example using sample code:

model = LogisticRegression(link_function="probit")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(model.evaluate(X_test, y_test))
model.save_model("logistic_model.pkl")
loaded_model = LogisticRegression.load_model("logistic_model.pkl")

'''

    
