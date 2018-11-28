import numpy as np
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier


def partitionData():
    X_train = np.genfromtxt("/homes/iws/guohaz/CSE546/Final_Project/train_final_OHE.csv", delimiter=",")
    X_test = np.genfromtxt("/homes/iws/guohaz/CSE546/Final_Project/test_final_OHE.csv", delimiter=",")
    X_train = X_train[1:, :]  # Excluse first row which is title names
    return X_train[:, :-1], X_train[:, -1], X_test[:, :-1], X_test[:, -1]  # X_train, Y_train, X_test, Y_test


X_train, y_train, X_test, y_test = partitionData()

for i in range(len(10)):
    model = KNeighborsClassifier(n_neighbors=(i + 1))
    # Train the model using the training sets
    model.fit(X_train, y_train)

    #Predict Output
    predicted= model.predict(X_test) # 0:Overcast, 2:Mild
    print(predicted)
    print(model.score(X_test, y_test))