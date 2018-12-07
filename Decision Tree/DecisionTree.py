import numpy as np
import pandas as pd
import random
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as ml
ml.use('Agg')


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
print(np.shape(X_train))

train_accuracy = []
validation_accuracy = []

finalTree = tree.DecisionTreeClassifier()
max_accuracy = 0
min_depth = 0
#pca
print("Decision Tree with out pca")
for i in tqdm(range(n)):
    clf = tree.DecisionTreeClassifier(max_depth=(i+1))
    clf.fit(X_train, y_train)
    train_accuracy.append(clf.score(X_train, y_train)*100)
    accuracy = clf.score(X_validation, y_validation)
    validation_accuracy.append(accuracy*100)
    tree.export_graphviz(clf, out_file='tree'+ str(i+1) + '.dot')
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        finalTree = clf
        min_depth = i+1
    
for i in range(n):
    print("depth : ", i+1)
    print("train error : ", train_accuracy[i])
    print("validation error : ", validation_accuracy[i])

print("+++++++++++++++++++++++++++++++++++++++++++++++")
print("best tree was depth ", min_depth)
print("test error : ", finalTree.score(X_test, y_test))
print("+++++++++++++++++++++++++++++++++++++++++++++++")



X_train, y_train, X_test, y_test = partitionData()

pca = PCA(n_components = 1000)
X_train_transformed = pca.fit_transform(X_train)
X_test_transformed = pca.transform(X_test)
train_accuracy_PCA = []
validation_accuracy_PCA = []
print(np.shape(X_train))
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


finalTree = tree.DecisionTreeClassifier()
max_accuracy_PCA = 0
min_depth_PCA = 0
#pca
print("Decision Tree with pca")
for i in tqdm(range(n)):
    clf = tree.DecisionTreeClassifier(max_depth=(i+1))
    clf.fit(X_train_transformed, y_train)
    train_accuracy_PCA.append(clf.score(X_train_transformed, y_train)*100)
    accuracy = clf.score(X_train_transformed_validation, y_validation)
    validation_accuracy_PCA.append(accuracy*100)
    tree.export_graphviz(clf, out_file='tree_PCA'+ str(i+1) + '.dot')
    if accuracy > max_accuracy_PCA:
        max_accuracy_PCA = accuracy
        finalTree = clf
        min_depth_PCA = i+1
    #M 256 n = 5 p 3

print("Decision Tree with out pca")
for i in range(n):
    print("depth : ", i+1)
    print("train error : ", train_accuracy_PCA[i])
    print("validation error : ", validation_accuracy_PCA[i])

print("+++++++++++++++++++++++++++++++++++++++++++++++")
print("best tree was depth ", min_depth)
print("test error : ", finalTree.score(X_test_transformed, y_test))
print("+++++++++++++++++++++++++++++++++++++++++++++++")


plt.title("Decision Tree accuracy with one hat")
plt.plot(range(1, 51), train_accuracy, label = 'train accuracy without PCA', color = 'aqua')
plt.plot(range(1, 51), validation_accuracy, label = 'validation accuracy without PCA', color = 'aquamarine')

plt.plot(range(1, 51), train_accuracy_PCA, label = 'train accuracy with PCA', color = 'orange')
plt.plot(range(1, 51), validation_accuracy_PCA, label = 'validation accuracy with PCA', color = 'orangered')

plt.scatter(min_depth, max_accuracy*100 , label = 'test accuracy witout PCA', color = 'blue')
plt.scatter(min_depth_PCA, max_accuracy_PCA*100 , label = 'test accuracy with PCA', color = 'brown')
plt.xlabel('depth of decision tree')
plt.xlabel('accuracy')

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.savefig('Decision_Tree_accuracy_hatted.png',bbox_inches='tight')
#plt.show()
plt.close()
