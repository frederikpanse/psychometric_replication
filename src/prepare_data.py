import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, test_size=0.3, random_state=3103, stratify=True):
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.scaler = StandardScaler()
    
    def preprocess(self, data):
        X = data.filter(like='X')  
        y = data['y_cat'] 
       
        y_one_hot = pd.get_dummies(y, prefix='y_cat')
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_one_hot, test_size=self.test_size, random_state=self.random_state, stratify=y if self.stratify else None
        )
        ### Ask Jeremy about standardizationa and stratification ###
        
        X_train_scaled = self.scaler.fit_transform(X_train)  
        X_test_scaled = self.scaler.transform(X_test)  
        
        return X_train_scaled, X_test_scaled, y_train, y_test


'''
You can use it for example by 
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.preprocess(data)
'''
