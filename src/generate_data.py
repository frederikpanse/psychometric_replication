import numpy as np
import pandas as pd
from itertools import product

def generate_data(n_classes_dep_var, n_ind_vars, n_categorical_vars, n_classes_ind_vars, n_samples):
    
    # set coefficients for the model
    coef = [3, 2, 2, 1, 1, 0.5, 0.5]
    coef = coef[:n_ind_vars]

    # generate n x p array for independent variables (predictors)
    X = np.random.uniform(0, 1, (n_samples, n_ind_vars))

    # convert independent continuous variables into categorical ones
    if n_categorical_vars == 0:
        if n_classes_ind_vars != 2:  # if no categorical vars, then do not vary number of classes
            return None
    elif n_categorical_vars >= n_ind_vars:
        return None
    else:
        for i in range(1, n_categorical_vars + 1):
            if n_classes_ind_vars == 2:
                X[:, -i] = np.where(X[:, -i] < np.quantile(X[:, -i], 0.5), 0, 1)  # 0 and 1 for binary
            elif n_classes_ind_vars == 3:
                X[:, -i] = np.where(X[:, -i] < np.quantile(X[:, -i], 0.25), 0,
                                    np.where(X[:, -i] > np.quantile(X[:, -i], 0.75), 2, 1))  # 0, 1, 2 for ternary
            
        

    # generate error term
    e = np.random.normal(0, 1, n_samples)

    # calculate y continuous
    y_continuous = 1 + np.dot(X, coef) + e

    # convert y to categorical
    # keep working from here
    if n_classes_dep_var == 2:
        y_categorical = np.where(y_continuous < np.quantile(y_continuous, 0.5), 'A', 'B')
    elif n_classes_dep_var == 3: 
        y_categorical = np.where(y_continuous < np.quantile(y_continuous, 0.25), 'A',
                                 np.where(y_continuous > np.quantile(y_continuous, 0.75), 'B', 'C'))
    elif n_classes_dep_var == 4: 
        y_categorical = np.where(y_continuous < np.quantile(y_continuous, 0.25), 'A',
                                 np.where(y_continuous < np.quantile(y_continuous, 0.5), 'B',
                                 np.where(y_continuous < np.quantile(y_continuous, 0.75), 'C', 'D')))

    # create data that is to be returned
    data_mat = np.concatenate((X, y_continuous.reshape(-1, 1), y_categorical.reshape(-1, 1)), axis = 1)
    data = pd.DataFrame(data_mat)
    # Rename columns to X1, X2, ..., y_cont, y_cat
    data.columns = [f'X{i+1}' for i in range(n_ind_vars)] + ['y_cont', 'y_cat']

    return(data)
