import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from itertools import product
from Logistic_Regression import LogisticRegression
from Decision_Tree import DecisionTree
from Random_Forest import RandomForest
from ANN_Pytorch import ANN
from prepare_data import DataPreprocessor
from generate_data import DataGenerator
from tqdm import tqdm


# Configurations
n_classes_dep_var = [2, 3, 4]
n_ind_vars = [3, 5, 7]
n_categorical_vars = np.arange(7)
n_classes_ind_vars = [2, 3]
n_samples = [100, 500, 1000, 10000]
splitting_criteria = ['entropy', 'gini', 'f_test']
min_samples_leaf_percents = [0.05, 0.1]
min_samples_split_percents = [0.1, 0.2]
learning_rate = 0.1
nLayer = [1,2]
nHidden = [3,15]
activation_function = nn.ReLU()
optimizer = ['SGD', 'Adam', 'RMSprop']
# Initialize preprocessor
preprocessor = DataPreprocessor()

# Initialize results list to store
results_list = []


total_models = (
    len(n_classes_dep_var) * len(n_ind_vars) * len(n_categorical_vars) * 
    len(n_classes_ind_vars) * len(n_samples) *
    (len(['logit', 'probit']) +  # Logistic Regression
     len(splitting_criteria) * len(min_samples_leaf_percents) * len(min_samples_split_percents) * 2 +  # Decision Tree and Random Forest
     len(nLayer) * len(nHidden))  # ANN
)

# Initialize tqdm progress bar to track the approximated progress
progress_bar = tqdm(total=total_models, desc="Training models")

# For computational reasons we perform 1 iteration, however, there is an option to do more
for iteration in range(1):
    print(f"Starting iteration {iteration + 1}...")

    for n_classes_dep, n_ind, n_categorical, n_classes_ind, n_samp in product(
        n_classes_dep_var, n_ind_vars, n_categorical_vars, n_classes_ind_vars, n_samples
    ):
        # Generate data
        data_generator = DataGenerator(
            n_classes_dep_var=n_classes_dep,
            n_ind_vars=n_ind,
            n_categorical_vars=n_categorical,
            n_classes_ind_vars=n_classes_ind,
            n_samples=n_samp
        )
        data = data_generator.generate_data()

        if data is None:
            continue

        # Preprocess data
        X_train, X_test, y_train, y_test = preprocessor.preprocess(data)
        nInput = X_train.shape[1]
        nOutput = len(set(y_train))

        # Logistic Regression
        for link_function in ['logit', 'probit']:
            model = LogisticRegression(link_function=link_function)
            model.fit(X_train, y_train)
            accuracy, precision, recall, f1 = model.evaluate(X_test, y_test)
            misclasification = 1 - accuracy
            results_list.append([
                iteration + 1, n_classes_dep, n_ind, n_categorical, n_classes_ind, n_samp,
                'LogisticRegression', link_function, None, None,misclasification, accuracy, precision, recall, f1
            ])
            progress_bar.update()

        # Decision Tree
        for criterion, min_samples_leaf_percent, min_samples_split_percent in product(
            splitting_criteria, min_samples_leaf_percents, min_samples_split_percents
        ):
            model = DecisionTree(
                criterion=criterion,
                min_samples_leaf=min_samples_leaf_percent,
                min_samples_split=min_samples_split_percent
            )
            model.fit(X_train, y_train)
            accuracy, precision, recall, f1 = model.evaluate(X_test, y_test)
            misclasification = 1 - accuracy
            results_list.append([
                iteration + 1, n_classes_dep, n_ind, n_categorical, n_classes_ind, n_samp,
                'DecisionTree', criterion, min_samples_leaf_percent, min_samples_split_percent, misclasification, accuracy, precision, recall, f1
            ])
            progress_bar.update()

        # Random Forest
        for criterion, min_samples_leaf_percent, min_samples_split_percent in product(
            splitting_criteria, min_samples_leaf_percents, min_samples_split_percents
        ):
            model = RandomForest(
                criterion=criterion,
                min_samples_leaf=min_samples_leaf_percent,
                min_samples_split=min_samples_split_percent
            )
            model.fit(X_train, y_train)
            accuracy, precision, recall, f1 = model.evaluate(X_test, y_test)
            misclasification = 1 - accuracy
            results_list.append([
                iteration + 1, n_classes_dep, n_ind, n_categorical, n_classes_ind, n_samp,
                'RandomForest', criterion, min_samples_leaf_percent, min_samples_split_percent, misclasification, accuracy, precision, recall, f1
            ])
            progress_bar.update()
        
        # ANN
        for layer, hidden, opt in product(nLayer, nHidden, optimizer):  
            model = ANN(nInput=nInput, nOutput=nOutput, nLayer=layer, nHidden=hidden, 
                        activation_function=activation_function, optimizer=opt,
                        learning_rate=learning_rate)
        
            epochs = 10
            batch_size = 64
            model.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

            accuracy, precision, recall, f1 = model.evaluate(X_test, y_test)
            misclasification = 1 - accuracy
            results_list.append([
                iteration + 1, n_classes_dep, n_ind, n_categorical, n_classes_ind, n_samp,
                'ANN', layer, hidden, opt, misclasification, accuracy, precision, recall, f1
            ])
            progress_bar.update()
        

        
# Save results to a DataFrame
results_df = pd.DataFrame(results_list, columns=[
    'Iteration', 'Classes_Dep_Var', 'Num_Ind_Vars', 'Num_Categorical_Vars',
    'Classes_Ind_Vars', 'Sample_Size', 'Model', 'Hyperparameter_1', 'Hyperparameter_2',
    'Hyperparameter_3', 'Miscalssification', 'Accuracy', 'Precision', 'Recall', 'F1_Score'
])

# Save results to a CSV file
results_df.to_csv('fixed_model_results.csv', index=False)
print("Results saved to fixed_model_results.csv.")
