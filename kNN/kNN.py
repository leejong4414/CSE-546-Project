import numpy as np
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier


def partitionData():
    print("process train")
    X_train = np.genfromtxt("/homes/iws/guohaz/CSE546/Final_Project/train_final_OHE.csv", delimiter=",", dtype = float)
    print("process test")
    X_test = np.genfromtxt("/homes/iws/guohaz/CSE546/Final_Project/test_final_OHE.csv", delimiter="," , dtype = float)
    print("finish process")
    X_train = X_train[1:, :]  # Excluse first row which is title names
    return X_train[1:, :-1], X_train[1:, -1], X_test[1:, :-1], X_test[1:, -1]  # X_train, Y_train, X_test, Y_test

X_train, y_train, X_test, y_test = partitionData()

for i in range(10):
    print(i)
    model = KNeighborsClassifier(n_neighbors=(i + 1))
    # Train the model using the training sets
    model.fit(X_train, y_train)

    #Predict Output
    predicted= model.predict(X_test)
    # print(predicted)
    print(model.score(X_test, y_test))