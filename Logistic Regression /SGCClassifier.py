from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import DataSource as ds

train_x, train_y, test_x, test_y = ds.partitionData()

hingeSGD = []
for i in range(50):
    clf2 = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001)
    clf2.fit(train_x, train_y)
    hingeSGD.append(clf2.score(test_x, test_y))
    if (i == 0):
        print(clf2)
print(np.mean(hingeSGD))



clf3 = RandomForestClassifier()
clf3.fit(train_x, train_y)
print(clf3.oob_score(test_x, test_y))



# squareSGD = []
# for i in range(50):
#     clf3 = SGDClassifier(loss='squared_loss', penalty='l1', alpha=0.0001)
#     clf3.fit(train_x, train_y)
#     squareSGD.append(clf3.score(test_x, test_y))
#     if (i == 0):
#         print(clf3)
# print(np.mean(squareSGD))

# logistCV = []
# for i in range(50):
#     logist = LogisticRegressionCV(cv=7, solver='sag', max_iter=10000)
#     logist.fit(train_x, train_y)
#     logistCV.append(logist.score(test_x, test_y))
#     if (i == 0):
#         print(logist)
# print(np.mean(logistCV))

# kNN = []
# for i in range(50):
#     kn = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
#     kn.fit(train_x, train_y)
#     kNN.append(kn.score(test_x, test_y))
#     if (i == 0):
#         print(kn)
# print(np.mean(kNN))




