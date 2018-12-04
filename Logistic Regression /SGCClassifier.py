from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import matplotlib as ml
ml.use('Agg')
import matplotlib.pyplot as plt

def partitionData():
    X_train = np.genfromtxt("./../Feature Engineering/train_final.csv", delimiter=",", dtype = float)
    X_test = np.genfromtxt("./../Feature Engineering/test_final.csv", delimiter=",", dtype = float)
    return X_train[1:,:-1], X_train[1:,-1], X_test[1:, :-1], X_test[1:,-1] #X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = partitionData()

cut = int (np.shape(X_train)[0] * 0.8)
print(cut)

X_Validation = X_train[cut:,:]
X_Train = X_train[:cut,:]
Y_Validation = Y_train[cut:]
Y_Train = Y_train[:cut]
n = 50

# bar_plt = {}
# bar_plt_v = {}
# # for i in range(50):
# clf2 = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001)
# clf2.fit(X_Train, Y_Train)
# print("Validation Error : {}".format(clf2.score(X_Validation, Y_Validation)))
# print("Test Error : {}".format(clf2.score(X_test, Y_test)))
# bar_plt_v['Hinge - L2'] = clf2.score(X_Validation, Y_Validation)
# bar_plt['Hinge - L2'] = clf2.score(X_test, Y_test)
# print(clf2)
#
# clf2 = SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.0001)
# clf2.fit(X_Train, Y_Train)
# print("Accuracy Validation: {}".format(clf2.score(X_Validation, Y_Validation)))
# print("Accuracy Test: {}".format(clf2.score(X_test, Y_test)))
# bar_plt_v['Squared - L2'] = clf2.score(X_Validation, Y_Validation)
# bar_plt['Squared - L2'] = clf2.score(X_test, Y_test)
# print(clf2)
#
# clf2 = SGDClassifier(loss='hinge', penalty='l1', alpha=0.0001)
# clf2.fit(X_Train, Y_Train)
# print("Accuracy Validation: {}".format(clf2.score(X_Validation, Y_Validation)))
# print("Accuracy Test: {}".format(clf2.score(X_test, Y_test)))
# bar_plt_v['Hinge - L1'] = clf2.score(X_Validation, Y_Validation)
# bar_plt['Hinge - L1'] = clf2.score(X_test, Y_test)
# print(clf2)
#
# clf2 = SGDClassifier(loss='squared_loss', penalty='l1', alpha=0.0001)
# clf2.fit(X_Train, Y_Train)
# print("Accuracy Validation: {}".format(clf2.score(X_Validation, Y_Validation)))
# print("Accuracy Test Data: {}".format(clf2.score(X_test, Y_test)))
# bar_plt_v['Squared - L1'] = clf2.score(X_Validation, Y_Validation)
# bar_plt['Squared L1'] = clf2.score(X_test, Y_test)
# print(clf2)
#
# plt.bar(list(bar_plt.keys()),list(bar_plt.values()))
# plt.title("Accuracy - Test")
# plt.xlabel("Loss Function and Penalty")
# plt.ylabel("Accuracy")
# plt.savefig("./test_AccLR.png")
# plt.close()
#
# plt.bar(list(bar_plt_v.keys()),list(bar_plt_v.values()))
# plt.title("Accuracy - Validatio ")
# plt.xlabel("Loss Function and Penalty")
# plt.ylabel("Accuracy")
# plt.savefig("./validation_AccLR.png")
# plt.close()



#     if (i == 0):
#         print(clf2)
# print(np.mean(hingeSGD))


# squareSGD = []
# for i in range(50):
#     clf3 = SGDClassifier(loss='squared_loss', penalty='l1', alpha=0.0001)
#     clf3.fit(train_x, train_y)
#     squareSGD.append(clf3.score(test_x, test_y))
#     if (i == 0):
#         print(clf3)
# print(np.mean(squareSGD))

# logistCV = []

# plot in terms of cv
# plot in terms of penalty
# plot in terms of
cv = range(2,6)
solvers = ['sag','saga','lbfgs']
penalty = ['l1','l2']
# for i in range(50):

logisc_t = {}
logisc_v = {}

for i in tqdm(cv):
    print("Training with CV = {}".format(i))
    for pen in tqdm(penalty):
        print("Penalty = {}".format(pen))
        logist = LogisticRegressionCV(cv=i, solver='saga', max_iter=10000, penalty=pen)
        logist.fit(X_train, Y_train)
        key = "Pen={},CV={}".format(pen,i)
        logisc_t[key] = logist.score(X_test, Y_test)
        logisc_v[key] = logist.score(X_Validation, Y_Validation)

plt.bar(list(logisc_t.keys()), list(logisc_t.values()))
plt.title("Accuracy Test")
plt.xlabel("Penalty with CV")
plt.ylabel("Accuracy")
plt.savefig("./LogisticTest.png")
plt.close()

plt.bar(list(logisc_v.keys()), list(logisc_v.values()))
plt.title("Accuracy Validation")
plt.xlabel("Penalty with CV")
plt.ylabel("Accuracy")
plt.savefig("./LogisticValidation.png")
plt.close()





print(logist.score(X_test, Y_test))
print(logist)
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
