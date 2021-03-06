#import matplotlib as ml
#ml.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA



import numpy as np
np.random.seed(10)


def partitionData():
    X_train = np.genfromtxt("./../Feature Engineering/train_final.csv", delimiter=",", dtype = float)
    X_test = np.genfromtxt("./../Feature Engineering/test_final.csv", delimiter=",", dtype = float)
    return X_train[1:,:-1], X_train[1:,-1], X_test[1:, :-1], X_test[1:,-1] #X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = partitionData()

clf = LogisticRegression(penalty='l1').fit(X_train, Y_train)
fprL1, tprL1, threshold = roc_curve(Y_test, clf.predict_proba(X_test)[:,1])
plt.plot([0,1],[0,1],'k--')
plt.plot(fprL1, tprL1, label="Logistic L1")
clf = LogisticRegression(penalty='l2').fit(X_train, Y_train)
fprL2, tprL2, threshold = roc_curve(Y_test, clf.predict_proba(X_test)[:,1])
plt.plot(fprL2, tprL2, label="Logistic L2")


## FIL IN CODE HERE ##
## 1. Train your model
## 2. Call predict_proba(X_test), check the documentations, it should have this method
## 3. fpr, tpr, threshold = roc_curve(Y_test, clf.predict_proba(X_test))
## 4. plot fpr on xaxis, tpr on y-axis and label your plot with the name of your model

# clf = tree.DecisionTreeClassifier(max_depth=25).fit(X_train, Y_train)
# fprD, tprD, threshold = roc_curve(Y_test, clf.predict_proba(X_test)[:,1])
# plt.plot(fprD, tprD, label="Decision Tree")
#
#
# model = KNeighborsClassifier(n_neighbors=(8))
# model.fit(X_train, Y_train)
# fprK, tprK, threshold = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
# plt.plot(fprK, tprK, label="kNN k=8")

#Predict Output
# predicted= model.predict(X_test_transformed)

pca = PCA(n_components=36)
x_transformed = pca.fit_transform(X_train)
xt_transformed = pca.transform(X_test)

cut = int(np.shape(x_transformed)[0] * 0.8)
X_Validation = x_transformed[cut:, :]
X_Train = x_transformed[:cut, :]
Y_Validation = Y_train[cut:]
Y_Train = Y_train[:cut]

clf = LogisticRegression(penalty='l1').fit(X_Train, Y_Train)
fprL1, tprL1, threshold = roc_curve(Y_test, clf.predict_proba(xt_transformed)[:,1])
plt.plot(fprL1, tprL1, label="PCA = 36 L1")


pca = PCA(n_components=13)
x_transformed = pca.fit_transform(X_train)
xt_transformed = pca.transform(X_test)

cut = int(np.shape(x_transformed)[0] * 0.8)
X_Validation = x_transformed[cut:, :]
X_Train = x_transformed[:cut, :]
Y_Validation = Y_train[cut:]
Y_Train = Y_train[:cut]

clf = LogisticRegression(penalty='l2').fit(X_Train, Y_Train)
fprL2, tprL2, threshold = roc_curve(Y_test, clf.predict_proba(xt_transformed)[:,1])
plt.plot(fprL2, tprL2, label="PCA = 13 L2")


plt.xlim()
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC(PCA).png")
plt.legend()
plt.show()
#plt.savefig("./ROC.png")


# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fprL2, tprL2, label="Logistic L2")
# plt.plot(fprL1, tprL1, label="Logistic L1")


#plt.legend()
#plt.savefig("./ROCZoom.png")
#plt.close()
