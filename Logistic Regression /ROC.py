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
print(clf.predict_proba(X_test).shape)
