import numpy as np
import pandas as pd
from itertools import product


class DataGenerator:
    def __init__(self, n_classes_dep_var, n_ind_vars, n_categorical_vars, n_classes_ind_vars, n_samples):
        self.n_classes_dep_var = n_classes_dep_var
        self.n_ind_vars = n_ind_vars
        self.n_categorical_vars = n_categorical_vars
        self.n_classes_ind_vars = n_classes_ind_vars
        self.n_samples = n_samples

        # Set coefficients for the model
        self.coef = [3, 2, 2, 1, 1, 0.5, 0.5]
        self.coef = self.coef[:self.n_ind_vars]

    def generate_data(self):
        # Generate n x p array for independent variables (predictors)
        X = np.random.uniform(0, 1, (self.n_samples, self.n_ind_vars))

        # Convert independent continuous variables into categorical ones
        if self.n_categorical_vars == 0:
            if self.n_classes_ind_vars != 2:  # If no categorical vars, then do not vary number of classes
                return None
        elif self.n_categorical_vars >= self.n_ind_vars:
            return None
        else:
            for i in range(1, self.n_categorical_vars + 1):
                if self.n_classes_ind_vars == 2:
                    X[:, -i] = np.where(X[:, -i] < np.quantile(X[:, -i], 0.5), 0, 1)  # 0 and 1 for binary
                elif self.n_classes_ind_vars == 3:
                    X[:, -i] = np.where(X[:, -i] < np.quantile(X[:, -i], 0.25), 0,
                                        np.where(X[:, -i] > np.quantile(X[:, -i], 0.75), 2, 1))  # 0, 1, 2 for ternary

        # Generate error term
        e = np.random.normal(0, 1, self.n_samples)

        # Calculate y continuous
        y_continuous = 1 + np.dot(X, self.coef) + e

        # Convert y to categorical
        if self.n_classes_dep_var == 2:
            y_categorical = np.where(y_continuous < np.quantile(y_continuous, 0.5), 'A', 'B')
        elif self.n_classes_dep_var == 3:
            y_categorical = np.where(y_continuous < np.quantile(y_continuous, 0.25), 'A',
                                     np.where(y_continuous > np.quantile(y_continuous, 0.75), 'B', 'C'))
        elif self.n_classes_dep_var == 4:
            y_categorical = np.where(y_continuous < np.quantile(y_continuous, 0.25), 'A',
                                     np.where(y_continuous < np.quantile(y_continuous, 0.5), 'B',
                                              np.where(y_continuous < np.quantile(y_continuous, 0.75), 'C', 'D')))

        # Create data to be returned
        data_mat = np.concatenate((X, y_continuous.reshape(-1, 1), y_categorical.reshape(-1, 1)), axis=1)
        data = pd.DataFrame(data_mat)
        
        # Rename columns to X1, X2, ..., y_cont, y_cat
        data.columns = [f'X{i+1}' for i in range(self.n_ind_vars)] + ['y_cont', 'y_cat']

        return data
