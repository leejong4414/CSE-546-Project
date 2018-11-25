import numpy as np
import pandas as pd
import random
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

def partitionData():
    X = np.genfromtxt("output_data.csv", delimiter=",")
    X = X[1:,:] # Excluse first row which is title names
    train_C = int(X.shape[0] * 0.8)
    train = X[:train_C,:]
    test = X[(train_C + 1):,:]
    return train[:,:-1], train[:,-1], test[:, :-1], test[:,-1] #X_train, Y_train, X_test, Y_test

X_train, y_train, X_test, y_test = partitionData()

#pca
print("Decision Tree")
for i in range(1,20):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf.fit(X_train, y_train)
    print(str(i) +" : " + str(clf.score(X_test, y_test)))
    #tree.export_graphviz(clf, out_file='tree'+ str(i) + '.dot')


print("Decision Tree with pca")
pca = PCA()
X_train_transformed = pca.fit_transform(X_train)
X_test_transformed = pca.transform(X_test)

for i in range(1,20):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf.fit(X_train_transformed, y_train)
    print(str(i) +" : " + str(clf.score(X_test_transformed, y_test)))
    #tree.export_graphviz(clf, out_file='tree'+ str(i) + '.dot')





# bg = BaggingClassifier()
# print(bg.fit(X_train, y_train))
# print(bg.score(X_test, y_test))

# bgb = GradientBoostingClassifier()
# print(bgb.fit(X_train, y_train))
# print(bgb.score(X_test, y_test))
