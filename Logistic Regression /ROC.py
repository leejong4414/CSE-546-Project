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

import numpy as np
np.random.seed(10)


def partitionData():
    X_train = np.genfromtxt("./../Feature Engineering/train_final.csv", delimiter=",", dtype = float)
    X_test = np.genfromtxt("./../Feature Engineering/test_final.csv", delimiter=",", dtype = float)
    return X_train[1:,:-1], X_train[1:,-1], X_test[1:, :-1], X_test[1:,-1] #X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = partitionData()

clf = LogisticRegression(penalty='l1').fit(X_train, Y_train)
fpr, tpr, threshold = roc_curve(Y_test, clf.predict_proba(X_test))
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr, label="Logistic L1")
clf = LogisticRegression(penalty='l2').fit(X_train, Y_train)
fpr, tpr, threshold = roc_curve(Y_test, clf.predict_proba(X_test))
plt.plot(fpr, tpr, label="Logistic L2")

## FIL IN CODE HERE ##
## 1. Train your model
## 2. Call predict_proba(X_test), check the documentations, it should have this method
## 3. fpr, tpr, threshold = roc_curve(Y_test, clf.predict_proba(X_test))
## 4. plot fpr on xaxis, tpr on y-axis and label your plot with the name of your model



plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC.png")
plt.show()
plt.savefig("./ROC.png")