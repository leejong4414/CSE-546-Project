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

# def partitionData():
#     X_train = pd.read_csv('./../Feature Engineering/train_final_OHE.csv')
#     X_test =  pd.read_csv('./../Feature Engineering/test_final_OHE.csv')
    
#     print(np.shape(X_train.as_matrix()))
#     X_train.dropna()
#     X_train = X_train.as_matrix()
#     print(np.shape(X_train))
    
#     print(np.shape(X_train.as_matrix()))
#     X_train.dropna()
#     X_test = X_test.as_matrix()
#     print(np.shape(X_test))
  
#     return X_train[:,:-1], X_train[:,-1], X_test[:, :-1], X_test[:,-1] #X_train, Y_train, X_test, Y_test

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


train_error = []
test_error = []

finalTree = tree.DecisionTreeClassifier()
min_error = float('inf')
min_depth = 0
#pca
print("Decision Tree with out pca")
for i in tqdm(range(n)):
    clf = tree.DecisionTreeClassifier(max_depth=(i+1))
    clf.fit(X_train, y_train)
    train_error.append(clf.score(X_train, y_train))
    error = clf.score(X_validation, y_validation)
    test_error.append(error)
    tree.export_graphviz(clf, out_file='tree'+ str(i) + '.dot')
    if error < min_error:
        finalTree = clf
        min_depth = i+1
    

print("Decision Tree with out pca")
for i in range(n):
    print("depth : ", i+1)
    print("train error : ", train_error[i])
    print("validation error : ", test_error[i])

print("best tree was depth ", min_depth)
print("test error : ", finalTree.score(X_test, y_test))


train_error = []
test_error = []

finalTree = tree.DecisionTreeClassifier()
min_error = float('inf')
min_depth = 0

print("Decision Tree with pca")
pca = PCA()
X_train_transformed = pca.fit_transform(X_train)
X_test_transformed = pca.transform(X_test)

for i in tqdm(range(n)):
    clf = tree.DecisionTreeClassifier(max_depth=(i+1))
    clf.fit(X_train_transformed, y_train)
    train_error.append(clf.score(X_train, y_train))
    test_error.append(clf.score(X_validation, y_validation))
    tree.export_graphviz(clf, out_file='tree_PCA'+ str(i) + '.dot')
    if error < min_error:
        finalTree = clf
        min_depth = i+1



print("Decision Tree with pca")
for i in range(n):
    print("depth : ", i+1)
    print("train error : ", train_error[i])
    print("validation error : ", test_error[i])

print("best tree was depth ", min_depth)
print("test error : ", finalTree.score(X_test, y_test))
    



# bg = BaggingClassifier()
# print(bg.fit(X_train, y_train))
# print(bg.score(X_test, y_test))

# bgb = GradientBoostingClassifier()
# print(bgb.fit(X_train, y_train))
# print(bgb.score(X_test, y_test))
