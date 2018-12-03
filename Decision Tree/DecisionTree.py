import numpy as np
import pandas as pd
import random
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from tqdm import tqdm

def partitionData():
    X_train = np.genfromtxt("./../Feature Engineering/train_final.csv", delimiter=",", dtype = float)
    X_test = np.genfromtxt("./../Feature Engineering/test_final.csv", delimiter=",", dtype = float)
    
    return X_train[1:,:-1], X_train[1:,-1], X_test[1:, :-1], X_test[1:,-1] #X_train, Y_train, X_test, Y_test

X_train, y_train, X_test, y_test = partitionData()
print(np.shape(X_train))
print(np.shape(y_train))
cut = int (np.shape(X_train)[0] * 0.8)
print(cut)

X_validation = X_train[cut:,:]
X_train = X_train[:cut,:]
y_validation = y_train[cut:]
y_train = y_train[:cut]
n = 50

# train_accuracy = []
# validation_accuracy = []

# finalTree = tree.DecisionTreeClassifier()
# max_accuracy = 0
# min_depth = 0
# #pca
# print("Decision Tree with out pca")
# for i in tqdm(range(n)):
#     clf = tree.DecisionTreeClassifier(max_depth=(i+1))
#     clf.fit(X_train, y_train)
#     train_accuracy.append(clf.score(X_train, y_train))
#     accuracy = clf.score(X_validation, y_validation)
#     validation_accuracy.append(accuracy)
#     tree.export_graphviz(clf, out_file='tree'+ str(i) + '.dot')
#     if accuracy > max_accuracy:
#         max_accuracy = accuracy
#         finalTree = clf
#         min_depth = i+1
    
# for i in range(n):
#     print("depth : ", i+1)
#     print("train error : ", train_accuracy[i])
#     print("validation error : ", validation_accuracy[i])

# print("+++++++++++++++++++++++++++++++++++++++++++++++")
# print("best tree was depth ", min_depth)
# print("test error : ", finalTree.score(X_test, y_test))
# print("+++++++++++++++++++++++++++++++++++++++++++++++")




pca = PCA()
X_train_transformed = pca.fit_transform(X_train)
X_test_transformed = pca.transform(X_test)
train_accuracy = []
validation_accuracy = []
# print("X_test_transformed" , np.shape(X_train_transformed))
# print("X_test_transformed", np.shape(X_test_transformed))

cut = int (np.shape(X_train_transformed)[0] * 0.8)
X_train_transformed_validation = X_train_transformed[cut:,:]
X_train_transformed = X_train_transformed[:cut,:]
y_validation = y_train[cut:]
y_train = y_train[:cut]
# print("X_train_transformed_validation",np.shape(X_train_transformed_validation))
# print("X_train_transformed",np.shape(X_train_transformed))
# print("y_validation",np.shape(y_validation))
# print("y_train",np.shape(y_train))
n = 50


finalTree = tree.DecisionTreeClassifier()
max_accuracy = 0
min_depth = 0
#pca
print("Decision Tree with pca")
for i in tqdm(range(n)):
    clf = tree.DecisionTreeClassifier(max_depth=(i+1))
    clf.fit(X_train_transformed, y_train)
    train_accuracy.append(clf.score(X_train_transformed, y_train))
    accuracy = clf.score(X_train_transformed_validation, y_validation)
    validation_accuracy.append(accuracy)
    tree.export_graphviz(clf, out_file='tree'+ str(i) + '.dot')
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        finalTree = clf
        min_depth = i+1
    

print("Decision Tree with out pca")
for i in range(n):
    print("depth : ", i+1)
    print("train error : ", train_accuracy[i])
    print("validation error : ", validation_accuracy[i])

print("+++++++++++++++++++++++++++++++++++++++++++++++")
print("best tree was depth ", min_depth)
print("test error : ", finalTree.score(X_test_transformed, y_test))
print("+++++++++++++++++++++++++++++++++++++++++++++++")