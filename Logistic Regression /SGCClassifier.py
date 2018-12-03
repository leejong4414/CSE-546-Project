from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def partitionData():
    X_train = np.genfromtxt("./../Feature Engineering/train_final.csv", delimiter=",", dtype = float)
    X_test = np.genfromtxt("./../Feature Engineering/test_final.csv", delimiter=",", dtype = float)
    return X_train[1:,:-1], X_train[1:,-1], X_test[1:, :-1], X_test[1:,-1] #X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = partitionData()


cut = int (np.shape(X_train)[0] * 0.8)
print(cut)

X_validation = X_train[cut:,:]
X_train = X_train[:cut,:]
y_validation = Y_train[cut:]
y_train = Y_train[:cut]
n = 50

# for i in range(50):
clf2 = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001)
clf2.fit(X_train, Y_train)
print("Validation Error : {}".format(clf2.score(X_validation, y_validation)))
print("Test Error : {}".format(clf2.score(X_test, Y_test)))
print(clf2)

clf2 = SGDClassifier(loss='squared', penalty='l2', alpha=0.0001)
clf2.fit(X_train, Y_train)
print("Validation Error : {}".format(clf2.score(X_validation, y_validation)))
print("Test Error : {}".format(clf2.score(X_test, Y_test)))
print(clf2)

clf2 = SGDClassifier(loss='hinge', penalty='l1', alpha=0.0001)
clf2.fit(X_train, Y_train)
print("Validation Error : {}".format(clf2.score(X_validation, y_validation)))
print("Test Error : {}".format(clf2.score(X_test, Y_test)))
print(clf2)

clf2 = SGDClassifier(loss='squared', penalty='l1', alpha=0.0001)
clf2.fit(X_train, Y_train)
print("Validation Error : {}".format(clf2.score(X_validation, y_validation)))
print("Test Error : {}".format(clf2.score(X_test, Y_test)))
print(clf2)

#     if (i == 0):
#         print(clf2)
# print(np.mean(hingeSGD))


# squareSGD = []
# for i in range(50):
#     clf3 = SGDClassifier(loss='squared_loss', penalty='l1', alpha=0.0001)
#     clf3.fit(train_x, train_y)
#     squareSGD.append(clf3.score(test_x, test_y))
#     if (i == 0):
#         print(clf3)
# print(np.mean(squareSGD))

# logistCV = []
# for i in range(50):
logist = LogisticRegressionCV(cv=4, solver='sag', max_iter=10000)
logist.fit(X_train, Y_train)
print(logist.score(X_test, Y_test))
print(logist)
#     if (i == 0):
#         print(logist)
# print(np.mean(logistCV))

# kNN = []
# for i in range(50):
#     kn = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
#     kn.fit(train_x, train_y)
#     kNN.append(kn.score(test_x, test_y))
#     if (i == 0):
#         print(kn)
# print(np.mean(kNN))
