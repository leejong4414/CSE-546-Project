import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

def partitionData():
        X = np.genfromtxt("output_data.csv", delimiter=",")
        X = X[1:,:] # Excluse first row which is title names
        train_C = int(X.shape[0] * 0.8)
        train = X[:train_C,:]
        test = X[(train_C + 1):,:]
        return train[:,:-1], train[:,-1], test[:, :-1], test[:,-1] #X_train, Y_train, X_test, Y_test

X_train, y_train, X_test, y_test = partitionData()

SGD = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001)

print(SGD.fit(X_train, y_train))
print(SGD.score(X_test, y_test))


SGD2 = SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.0001)

print(SGD2.fit(X_train, y_train))
print(SGD2.score(X_test, y_test))

X_train, y_train, X_test, y_test = partitionData()
clf = SVC(gamma='auto')
print(clf.fit(X_train, y_train))
print(clf.score(X_test, y_test))


clf2 = SVC(gamma='auto', probability = True)
print(clf2.fit(X_train, y_train))
print(clf2.score(X_test, y_test))



clf2 = SVC(gamma='auto', decision_function_shape='ovo')
print(clf2.fit(X_train, y_train))
print(clf2.score(X_test, y_test))

clf3 = SVC(kernel='linear')
print(clf3.fit(X_train, y_train))
print(clf3.score(X_test, y_test))

clf4 = SVC(kernel='linear', decision_function_shape='ovo')
print(clf4.fit(X_train, y_train))
print(clf4.score(X_test, y_test))

clf5 = SVC(kernel='poly')
print(clf5.fit(X_train, y_train))
print(clf5.score(X_test, y_test))

clf6 = SVC(kernel='sigmoid')
print(clf6.fit(X_train, y_train))
print(clf6.score(X_test, y_test))

clf7 = SVC(kernel='precomputed')
print(clf7.fit(X_train, y_train))
print(clf7.score(X_test, y_test))

clf8 = SVC(kernel='linear', decision_function_shape='ovo')
print(clf8.fit(X_train, y_train))
print(clf8.score(X_test, y_test))