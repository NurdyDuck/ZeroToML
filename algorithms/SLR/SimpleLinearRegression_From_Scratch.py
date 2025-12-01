import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

class MyLR:
    def __init__(self):
        self.m = None
        self.b = None
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        num = 0
        den = 0
        x_mean = X_train.mean()
        y_mean = y_train.mean()

        for i in range(X_train.shape[0]):
            num += ((X_train[i] - x_mean)*(y_train[i] - y_mean))
            den += (X_train[i] - x_mean)**2 
        if den == 0:
            print("Error: Cannot fit line because all X values are same.")
            return
        self.m = num/den
        self.b = y_mean - (self.m * x_mean)
    
    def predict(self, X_test):
        return (self.m * X_test) + self.b

df = pd.read_csv(r'C:\Users\DELL\Desktop\Python_child\placement.csv')
print(df.head())


X = df.iloc[:, 0].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

lr = MyLR()
lr.fit(X_train, y_train)

print(X_test[0])

print("Prediction: ", lr.predict(X_test[0]))
