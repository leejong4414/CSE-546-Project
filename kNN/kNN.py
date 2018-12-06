import numpy as np
import pandas as pd
import random
from sklearn import neighbors, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def partitionData():
    print("process train")
    X_train = np.genfromtxt("/homes/iws/guohaz/CSE546/Final_Project/train_processed.csv", delimiter=",", dtype = float)
    # X_train = np.genfromtxt("./train_processed.csv", delimiter=",", dtype=float)
    # print("process test")
    # X_test = np.genfromtxt("/homes/iws/guohaz/CSE546/Final_Project/test_final.csv", delimiter="," , dtype = float)
    print("finish process")
    X_train = X_train[1:, :]  # Excluse first row which is title names
    return X_train[1:, :-1], X_train[1:, -1]#, X_test[1:, :-1], X_test[1:, -1]  # X_train, Y_train, X_test, Y_test

X_train, y_train = partitionData() #, X_test, y_test = partitionData()

# pca = PCA()
# X_train_transformed = pca.fit_transform(X_train)
# X_test_transformed = pca.transform(X_test)
#
# for i in tqdm(range(20)):
#     print(i)
#     model = KNeighborsClassifier(n_neighbors=(i + 1))
#     # Train the model using the training sets
#     # model.fit(X_train, y_train)
#     model.fit(X_train_transformed, y_train)
#
#     #Predict Output
#     # predicted= model.predict(X_test_transformed)
#     # print(predicted)
#     print(model.score(X_test_transformed, y_test))
X = X_train[:, :2]
n_neighbors = 15
h = .02  # step size in the mesh
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('App id')
    plt.ylabel('Device id')
    plt.title("kNN classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.savefig("{}.png".format("kNN"))

