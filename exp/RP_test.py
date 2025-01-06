import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

# Load dataset from GitHub
def load_github_data(url):
    """
    Loads a CSV dataset from a GitHub raw URL.
    Args:
        url (str): The raw GitHub URL to the dataset.
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    try:
        data = pd.read_csv(url)
        print("Dataset successfully loaded from GitHub.")
        return data
    except Exception as e:
        print("Error loading dataset:", e)
        return None

# Preprocess data
def preprocess_data(data):
    """
    Prepares features and target for training.
    Assumes features are in columns `x1`, `x2`, `x3` and the target is `y_categorical`.
    """
    X = data[['x1', 'x2', 'x3']].values
    y = data['y_categorical'].values

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode labels if necessary (already binary in this example)
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y

# Logistic Regression
def logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy:", accuracy)

# Decision Tree
def decision_tree(X_train, y_train, X_test, y_test):
    sample_size = len(X_train)
    min_samples_leaf = int(0.05 * sample_size)
    min_samples_split = int(0.10 * sample_size)
    model = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Decision Tree Accuracy:", accuracy)

# Random Forest
def random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Accuracy:", accuracy)

# Neural Network (TensorFlow)
def neural_network(X_train, y_train, X_test, y_test):
    # Define the neural network model
    input_size = X_train.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, activation='relu', input_shape=(input_size,)),  # First hidden layer with 3 neurons
        tf.keras.layers.Dense(3, activation='relu'),                            # Second hidden layer with 3 neurons
        tf.keras.layers.Dense(1, activation='sigmoid')                          # Output layer for binary classification
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Neural Network Test Accuracy:", test_accuracy)

    # Predictions and accuracy
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print("Sklearn Neural Network Accuracy:", accuracy)

# Main function
def main():
    # URL to the dataset on GitHub
    github_url = "https://raw.githubusercontent.com/frederikpanse/psychometric_replication/refs/heads/main/dat/first_data.csv?token=GHSAT0AAAAAAC2NZ2ELTEKZVSAU7ZWLDY2YZ2HSXRQ"
    data = load_github_data(github_url)

    if data is not None:
        # Ensure dataset format matches expected columns
        if {'x1', 'x2', 'x3', 'y_categorical'}.issubset(data.columns):
            X, y = preprocess_data(data)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            print("Running Logistic Regression...")
            logistic_regression(X_train, y_train, X_test, y_test)

            print("\nRunning Decision Tree...")
            decision_tree(X_train, y_train, X_test, y_test)

            print("\nRunning Random Forest...")
            random_forest(X_train, y_train, X_test, y_test)

            print("\nRunning Neural Network (TensorFlow)...")
            neural_network(X_train, y_train, X_test, y_test)
        else:
            print("Error: Dataset does not contain the required columns.")
    else:
        print("Failed to load dataset from GitHub.")

if __name__ == "__main__":
    main()
