import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from itertools import product

#  Add from src when uploading to github?

from Logistic_Regression import LogisticRegression
from Decision_Tree import DecisionTree
from Random_Forest import RandomForest
from ANN_Pytorch import ANN
from prepare_data import DataPreprocessor
from generate_data import DataGenerator


'''
PLAN:
1. do 100 iterations
    2. for each iter:
        3. generate data 
            - generate using the following criteria: 
                n_classes_dep_var = [2, 3, 4]
                n_ind_vars = [3, 5, 7]
                n_categorical_vars = np.arange(7)
                n_classes_ind_vars = [2, 3]
                n_samples = [100, 500, 1000, 10000]
                sample loop: 
                np.random.seed(3103)
                for n_classes_dep, n_ind, n_categorical, n_classes_ind, n_samp in product(n_classes_dep_var, n_ind_vars, n_categorical_vars, n_classes_ind_vars, n_samples):
                    data = generate_data(n_classes_dep, n_ind, n_categorical, n_classes_ind, n_samp)
                    if data is not None:  # Avoid saving if no data is generated
                        filename = f'../dat/{n_categorical}_{n_classes_ind}_{n_ind}_{n_classes_dep}_{n_samp}_samples.csv'
                        data.to_csv(filename, index=False)
        4. preprcess data
        5. do cross-validation for each of the models and save the results of each model in a dictioanry or table that later can be downlaoded for example on github. 
            Logistic Regression should be trained for these params: 
                link_function = ['Probit', 'Logit']
            Decision tree should be trained for these params: 
                criteria = ['entropy', 'gini', 'f_test']  # Equivalent to 'Entropy reduction' and 'Gini reduction'
                min_samples_leaf = [0.05, 0.1]  # 5% and 10% of sample size
                min_samples_split = [0.1, 0.2]  # 10% and 20% of sample size
            Random Forest should be trained for these params:
                criteria = ['entropy', 'gini', 'f_test']  # Equivalent to 'Entropy reduction' and 'Gini reduction'
                min_samples_leaf = [0.05, 0.1]  # 5% and 10% of sample size
                min_samples_split = [0.1, 0.2]  # 10% and 20% of sample size
            The sample loop should look similar to this:
                results_list = []
# loop over all data sets
np.random.seed(3103)
idx = 0
for n_classes_dep, n_ind, n_categorical, n_classes_ind, n_samp in product(n_classes_dep_var, n_ind_vars, n_categorical_vars, n_classes_ind_vars, n_samples):
    data_generator = DataGenerator(
        n_classes_dep_var=n_classes_dep, 
        n_ind_vars=n_ind,        
        n_categorical_vars=n_categorical,
        n_classes_ind_vars=n_classes_ind,
        n_samples=n_samp         
    )
    data = data_generator.generate_data()
    idx += 1
    # data = generate_data(n_classes_dep, n_ind, n_categorical, n_classes_ind, n_samp)
    if data is not None:  # Avoid using if no data is generated 
        X_train, X_test, y_train, y_test = preprocessor.preprocess(data)

        for criterion, min_samples_leaf_percent, min_samples_split_percent in product(
            splitting_criteria, min_samples_leaf_percents, min_samples_split_percents
        ):
            min_samples_leaf = int(min_samples_leaf_percent * n_samp)
            min_samples_split = int(min_samples_split_percent * n_samp)

            clf = tree.DecisionTreeClassifier(
                criterion=criterion,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split
            )
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            misclassification = 1 - accuracy_score(y_test, y_pred)
            
            # Append results for this configuration
            results_list.append([
                f'S{idx}',                 # Simulation ID
                n_classes_dep,             # Dependent variable classes
                n_ind,                     # Number of independent variables
                n_categorical,             # Number of categorical variables
                n_classes_ind,             # Categorical independent variable classes
                n_samp,                    # Sample size
                criterion,                 # Splitting criterion
                min_samples_leaf_percent,  # Minimum samples per leaf (percent)
                min_samples_split_percent, # Minimum samples for split (percent)
                misclassification          # Misclassification rate calculated as 1 - accuracy_score
                Accuracy,
                Precision,
                Recall,
                F1_Score
            ])
            # Convert results to a DataFrame for analysis
results_df = pd.DataFrame(results_list, columns=[
    'Simulation_ID', 
    'Classes_Dep_Var', 
    'Num_Ind_Vars', 
    'Num_Categorical_Vars', 
    'Classes_Ind_Vars', 
    'Sample_Size', 
    'Splitting_Criterion', 
    'Min_Samples_Leaf_Percent', 
    'Min_Samples_Split_Percent', 
    'Misclassification',
    'Accuracy',
    'Precision',
    'Recall',
    'F1_Score'
])



The final table/dictionary should include the above columns 
            
        


'''


# Configuration
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

# Initialize results storage
results_list = []

# Calculate total models, get rid of it later
total_models = len(n_classes_dep_var) * len(n_ind_vars) * len(n_categorical_vars) * len(n_classes_ind_vars) * len(n_samples)
total_models *= (len(['logit', 'probit']) +  # Logistic Regression
                 len(splitting_criteria) * len(min_samples_leaf_percents) * len(min_samples_split_percents) * 2 +  # Decision Tree and Random Forest
                 len(nLayer) * len(nHidden))  # ANN
model_count = 0

# Perform 100 iterations
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
        nOutput = 1

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
            model_count += 1
            if model_count % 50 == 0:
                print(f"{model_count}/{total_models} models trained.")

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
            model_count += 1
            if model_count % 50 == 0:
                print(f"{model_count}/{total_models} models trained.")

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
            model_count += 1
            if model_count % 50 == 0:
                print(f"{model_count}/{total_models} models trained.")
        
        # ANN
        for layer, hidden, opt in product(nLayer, nHidden, optimizer):  # Fixed variable names
            model = ANN(nInput=nInput, nOutput=nOutput, nLayer=layer, nHidden=hidden,   # Used `layer`, `hidden`, `opt`
                        activation_function=activation_function, optimizer=opt,
                        learning_rate=learning_rate)
        
            epochs = 20
            batch_size = 32
            model.train(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)

            accuracy, precision, recall, f1 = model.evaluate(X_test, y_test)
            misclasification = 1 - accuracy
            results_list.append([
                iteration + 1, n_classes_dep, n_ind, n_categorical, n_classes_ind, n_samp,
                'ANN', layer, hidden, opt, misclasification, accuracy, precision, recall, f1
            ])
            model_count += 1
            if model_count % 50 == 0:
                print(f"{model_count}/{total_models} models trained.")
        

        
# Save results to a DataFrame
results_df = pd.DataFrame(results_list, columns=[
    'Iteration', 'Classes_Dep_Var', 'Num_Ind_Vars', 'Num_Categorical_Vars',
    'Classes_Ind_Vars', 'Sample_Size', 'Model', 'Hyperparameter_1', 'Hyperparameter_2',
    'Hyperparameter_3', 'Miscalssification', 'Accuracy', 'Precision', 'Recall', 'F1_Score'
])

# Save results to a CSV file
results_df.to_csv('TESTING_model_results.csv', index=False)
print("Results saved to model_results.csv.")
