import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

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

clf = tree.DecisionTreeClassifier(max_depth=25).fit(X_train, Y_train)
fprD, tprD, threshold = roc_curve(Y_test, clf.predict_proba(X_test)[:,1])
plt.plot(fprD, tprD, label="Decision Tree")


model = KNeighborsClassifier(n_neighbors=(8))
model.fit(X_train, Y_train)
fprK, tprK, threshold = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
plt.plot(fprK, tprK, label="kNN k=8")

#Predict Output
# predicted= model.predict(X_test_transformed)



plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC.png")
plt.legend()
plt.show()
plt.savefig("./ROC.png")
plt.close()


plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fprD, tprD, label="Decision Tree")
plt.plot(fprK, tprK, label="kNN k=8")
plt.plot(fprL2, tprL2, label="Logistic L2")
plt.plot(fprL1, tprL1, label="Logistic L1")
plt.legend()
plt.show()
plt.savefig("./ROCZoom.png")
