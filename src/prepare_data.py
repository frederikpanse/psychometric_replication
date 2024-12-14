import pandas as pd
from sklearn.model_selection import train_test_split

# create function to pre process data
def prepare_data(data): 
    X = data.filter(like='X') 
    y = data['y_cat'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3103, stratify=y)
    return X_train, X_test, y_train, y_test
