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
    return X_train[:,:-1], X_train[:,-1], X_test[:, :-1], X_test[:,-1] #X_train, Y_train, X_test, Y_test

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



#pca
print("Decision Tree")
for i in tqdm(range(1,20)):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf.fit(X_train, y_train)
    print(str(i) +" : " + str(clf.score(X_train, y_train)))
    print(str(i) +" : " + str(clf.score(X_test, y_test)))
    #tree.export_graphviz(clf, out_file='tree'+ str(i) + '.dot')


# print("Decision Tree with pca")
# pca = PCA()
# X_train_transformed = pca.fit_transform(X_train)
# X_test_transformed = pca.transform(X_test)

# for i in range(1,20):
#     clf = tree.DecisionTreeClassifier(max_depth=i)
#     clf.fit(X_train_transformed, y_train)
#     print(str(i) +" : " + str(clf.score(X_train_transformed, y_train)))
#     print(str(i) +" : " + str(clf.score(X_test_transformed, y_test)))
#     #tree.export_graphviz(clf, out_file='tree'+ str(i) + '.dot')





# bg = BaggingClassifier()
# print(bg.fit(X_train, y_train))
# print(bg.score(X_test, y_test))

# bgb = GradientBoostingClassifier()
# print(bgb.fit(X_train, y_train))
# print(bgb.score(X_test, y_test))
