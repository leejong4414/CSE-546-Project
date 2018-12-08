import numpy as np
import pandas as pd
import random
from sklearn.model_selection import cross_val_score
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib as ml
ml.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def partitionData():
    print("process train")
    X_train = np.genfromtxt("/homes/iws/guohaz/CSE546/Final_Project/train_final.csv", delimiter=",", dtype = float)
    # X_train = np.genfromtxt("train_final.csv", delimiter=",", dtype=float)
    # X_train = np.genfromtxt("./train_processed.csv", delimiter=",", dtype=float)
    print("process test")
    X_test = np.genfromtxt("/homes/iws/guohaz/CSE546/Final_Project/test_final.csv", delimiter="," , dtype = float)
    # X_test = np.genfromtxt("test_final.csv", delimiter=",", dtype=float)
    print("finish process")
    X_train = X_train[1:, :]  # Excluse first row which is title names
    return X_train[1:, :-1], X_train[1:, -1], X_test[1:, :-1], X_test[1:, -1]  # X_train, Y_train, X_test, Y_test

X_train, y_train, X_test, y_test = partitionData()


myList = list(range(1, 20))
# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

for i in neighbors:
    print(i)
    model = KNeighborsClassifier(n_neighbors=i)#, weights='distance')
    # Train the model using the training sets
    model.fit(X_train, y_train)
    # model.fit(X_train_transformed, y_train)

    #Predict Output
    # predicted= model.predict(X_test)
    # print(predicted)
    print(model.score(X_train, y_train))
    print(model.score(X_test, y_test))

# empty list that will hold cv scores
# cv_scores = []
# dimension = [50, 100, 250, 500, 750]
# pca250 = PCA(n_components = 1)
# X_train_transformed = pca250.fit_transform(X_train)
# X_test_transformed = pca250.transform(X_test)
# perform 10-fold cross validation
# for k in neighbors:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     print(250, k)
#     scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
#     totalscore = scores.mean()
#     print(totalscore)
    # cv_scores.append(totalscore)


# pca500 = PCA(n_components = 500)
# X_train_transformed = pca500.fit_transform(X_train)
# X_test_transformed = pca500.transform(X_test)
# # perform 10-fold cross validation
# for k in neighbors:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     print(500, k)
#     scores = cross_val_score(knn, X_train_transformed, y_train, cv=10, scoring='accuracy')
#     totalscore = scores.mean()
#     print(totalscore)
# pca750 = PCA(n_components = 750)
# X_train_transformed = pca500.fit_transform(X_train)
# X_test_transformed = pca500.transform(X_test)
# # perform 10-fold cross validation
# for k in neighbors:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     print(750, k)
#     scores = cross_val_score(knn, X_train_transformed, y_train, cv=10, scoring='accuracy')
#     totalscore = scores.mean()
#     print(totalscore)

# changing to misclassification error
# MSE = [1 - x for x in cv_scores]

# determining best k
# optimal_k = neighbors[MSE.index(min(MSE))]
# print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
# plt.plot(neighbors, MSE)
# plt.xlabel('Number of Neighbors K')
# plt.ylabel('Misclassification Error')
# plt.savefig("{}.png".format("kNN"))
