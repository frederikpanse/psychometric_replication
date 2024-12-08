import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.metrics import accuracy_score
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
        self.criterion = nn.CrossEntropyLoss() if nOutput > 1 else nn.BCELoss()

    def load_data_from_csv(file_path):
        # Load data from csv file, later update so it get it from github url,  
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1].values  # Features (all columns except the last one)
        y = df.iloc[:, -1].values   # Target (the last column)
        # Scale features
        #scaler = StandardScaler()
        #X = scaler.fit_transform(X)

    # Encode labels if necessary (already binary in this example)
        le = LabelEncoder()
        y = le.fit_transform(y)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)  # For classification

        return X_tensor, y_tensor

        ###   Need to think how to do it with github   ####




    def _build_model(self):
        layers = []
        layers.append(nn.Linear(self.nInput, self.nHidden))
        layers.append(self.activation_function())
       # layers = [nn.Linear(self.nInput, self.nHidden), self.act_fn()]       ## Alternative to layers = []
        for layer in range(self.nLayer - 1):
            layers.append(nn.Linear(self.nHidden, self.nHidden))
            layers.append(self.activation_function())
        layers.append(nn.Linear(self.nHidden, self.nOutput))
        # Output activation for multi-class classification
        if self.nOutput > 1:
            layers.append(nn.Softmax(dim=1))
        else:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def _choose_optimizer(self):
        # Different optimizer options, we can test different if Jeremy says it's a good idea. 
        # Maybe add SGD with Momentum or Nesterov Momentum. Maybe don't test all of them might be too much?
        if self.optimizer_type == 'SGD':
            return optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif self.optimizer_type == 'Adam':
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'RMSprop':
            return optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'AdaGrad':
            return optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'AdaDelta':
            return optim.Adadelta(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"We didn't add this optimizer to the list: {self.optimizer_type}")


    def train(self, train_data, train_labels, epochs=10, batch_size=32):
        
        dataset = TensorDataset(train_data, train_labels)
        train_size = int(len(dataset) * 0.7)  # Use 70% for training
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            test_loss, test_accuracy = self._evaluate(test_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")

# In the above fucntion, should it divide the dataset now or will we do it before and then this fucntion shoould take train_data and test_data as arguments?


    def _evaluate(self, data_loader):
        # Function used inside train loop to evaluate the performance live during the training
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                total += targets.size(0)
                if self.nOutput > 1:
                    loss = self.criterion(outputs, targets)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                else:
                    loss = self.criterion(outputs.squeeze(), targets.float())
                    predicted = (outputs.squeeze() > 0.5).int()
                    correct += (predicted == targets).sum().item()
                total_loss += loss.item()

        accuracy = correct / total * 100
        return total_loss / len(data_loader), accuracy
    

    ###### Check A#4 Geiger's functions for test and train, might be better to use them

    def evaluate(self, test_data, test_labels):
    
        test_dataset = TensorDataset(test_data, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=32)
        test_loss, test_accuracy = self._evaluate(test_loader)
        print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")
        return test_loss, test_accuracy

    def sklearn_evaluation(self, test_data, test_labels):
        # Might be a better choice if we want to measure other performance metrics 
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(test_data)
            if self.nOutput > 1:
                _, predicted = torch.max(outputs, 1)
            else:
                predicted = (outputs.squeeze() > 0.5).int()

        accuracy = accuracy_score(test_labels, predicted)
        print(f"Sklearn Accuracy: {accuracy:.4f}")
        return accuracy
    

    def save_model(self, save_path):
        # Saves model so it can later be reused, change it so it saves it in github folder
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, save_path):
        # We will use this function to load previously trained models 
        self.model.load_state_dict(torch.load(save_path))
        print(f"Model loaded from {save_path}")


    ### Testing if the above works ###

    # Example of how to call fucntions

    def test_ann_with_csv(file_path):
    # Load and preprocess the data
        X_data, y_data = ANN.load_data_from_csv(file_path)
            
            # Define input and output dimensions, hidden layers, activation function, and optimizer
        nInput = X_data.shape[1]  # Number of features in input data
        nOutput = len(torch.unique(y_data))  # Number of unique labels in target
        nLayer = 3  # Number of hidden layers
        nHidden = 10  # Number of neurons in hidden layers
        activation_function = nn.ReLU  # Activation function
        optimizer = 'Adam'  # Optimizer type
            
            # Initialize the model
        model = ANN(nInput=nInput, nOutput=nOutput, nLayer=nLayer, nHidden=nHidden,
                        activation_function=activation_function, optimizer=optimizer, learning_rate=0.001)

            # Train the model
        print("Training the model...")
        model.train(X_data, y_data, epochs=10, batch_size=32)

            # Evaluate the model on test data
        print("\nEvaluating on test data...")
        model.evaluate(X_data, y_data)

            # Optionally, you can use sklearn evaluation
        print("\nSklearn Evaluation...")
        model.sklearn_evaluation(X_data, y_data)

# Test the model with a CSV file (replace with your actual file path)
#csv_file_path = "C:/Users/cypri/OneDrive/Desktop/TUB/Research Project - Google/code/first_data.csv"  # Update this path
#ANN.test_ann_with_csv(csv_file_path)