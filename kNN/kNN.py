import numpy as np
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier


def partitionData():
    X = np.genfromtxt("output_data.csv", delimiter=",")
    X = X[1:, :]  # Excluse first row which is title names
    train_C = int(X.shape[0] * 0.2)
    train = X[:train_C, :]
    test = X[(train_C + 1):, :]
    return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]  # X_train, Y_train, X_test, Y_test


X_train, y_train, X_test, y_test = partitionData()

model = KNeighborsClassifier(n_neighbors=3)
# Train the model using the training sets
model.fit(X_train, y_train)

#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print(predicted)