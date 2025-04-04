import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder



class ANN(nn.Module): 
    def __init__(self, nInput, nOutput, nLayer, nHidden, activation_function, optimizer, learning_rate=0.1, momentum=0.9):
        super(ANN, self).__init__()
        # Initialize all necessary arguments 
        self.nInput = nInput
        self.nOutput = nOutput
        self.nLayer = nLayer
        self.nHidden = nHidden
        self.activation_function= activation_function
        self.optimizer_type = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model = self._build_model()
        self.optimizer = self._choose_optimizer()
        #For multiclass classification we use Cross Entropy Loss, for the binary classification case we use Binary Cross Entropy Loss
        self.criterion = nn.CrossEntropyLoss() if nOutput > 1 else nn.BCEWithLogitsLoss()


    
    def _build_model(self):
        layers = []
        layers.append(nn.Linear(self.nInput, self.nHidden))
        layers.append(self.activation_function)
        for layer in range(self.nLayer - 1):
            layers.append(nn.Linear(self.nHidden, self.nHidden))
            layers.append(self.activation_function)
        layers.append(nn.Linear(self.nHidden, self.nOutput))
        # Output activation for multi-class classification
        if self.nOutput > 1:
            layers.append(nn.Softmax(dim=1))
        else:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _choose_optimizer(self):
        # Different optimizer options 
        if self.optimizer_type == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum,weight_decay=1e-4)
        elif self.optimizer_type == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        elif self.optimizer_type == 'RMSprop':
            return optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        else:
            raise ValueError(f"We didn't add this optimizer to the list: {self.optimizer_type}")


    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        
        # Convert to PyTorch tensors if necessary
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.tensor(X_test, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.float32 if y_train.ndim == 1 else torch.long)
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, dtype=torch.float32 if y_test.ndim == 1 else torch.long)

        # Create DataLoader for training data
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        patience = 3  # Number of epochs to wait before stopping
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets.long())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            test_loss, test_accuracy = self._evaluate(X_test, y_test)
            #print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%") #Can be used to track the performance

        # Early Stopping Logic
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    #print("Early stopping triggered.")
                    break

    def _evaluate(self, X_test, y_test):
    
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            if self.nOutput > 1:
            # Convert y_test to LongTensor
                loss = self.criterion(outputs, y_test.long())
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == y_test).sum().item()
            else:
                loss = self.criterion(outputs.squeeze(), y_test)
                predicted = (outputs.squeeze() > 0.5).int()
                correct = (predicted == y_test).sum().item()
            accuracy = correct / len(y_test) * 100
        return loss.item(), accuracy


    def evaluate(self, test_data, test_labels):
    # Convert to PyTorch tensors if necessary
        if not isinstance(test_data, torch.Tensor):
            test_data = torch.tensor(test_data, dtype=torch.float32)
        if not isinstance(test_labels, torch.Tensor):
            test_labels = torch.tensor(test_labels, dtype=torch.float32 if test_labels.ndim == 1 else torch.long)
    
    # Set the model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(test_data)
            if self.nOutput > 1:
                _, predicted = torch.max(outputs, 1)
            else:
                predicted = (outputs.squeeze() > 0.5).int()

        accuracy = accuracy_score(test_labels.numpy(), predicted.numpy())
        precision = precision_score(test_labels.numpy(), predicted.numpy(), average='weighted', zero_division=0)
        recall = recall_score(test_labels.numpy(), predicted.numpy(), average='weighted', zero_division=0)
        f1 = f1_score(test_labels.numpy(), predicted.numpy(), average='weighted', zero_division=0)
        return accuracy, precision, recall, f1
    

    def save_model(self, save_path):
        # Saves model so it can later be reused
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, save_path):
        # We will use this function to load previously trained models 
        self.model.load_state_dict(torch.load(save_path))
        print(f"Model loaded from {save_path}")

'''
You can use this class for example using the following code

nInput = X_train.shape[1]  # Number of input features
nOutput = 1  # Binary classification (output is 1 for binary)
nLayer = 2  # Number of hidden layers
nHidden = 32  # Number of neurons in each hidden layer
activation_function = nn.ReLU  # Activation function for hidden layers
optimizer = 'Adam'  # Optimizer
learning_rate = 0.001  # Learning rate

# Create the model
model = ANN(nInput=nInput, nOutput=nOutput, nLayer=nLayer, nHidden=nHidden,
            activation_function=activation_function, optimizer=optimizer,
            learning_rate=learning_rate)

# Train the model
epochs = 20
batch_size = 32
model.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

# Evaluate the model
print("\nEvaluating the model:")
model.sklearn_evaluation(X_test, y_test_)
'''
