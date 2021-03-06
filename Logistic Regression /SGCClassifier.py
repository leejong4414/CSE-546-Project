from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from tqdm import tqdm
import numpy as np
import matplotlib as ml
ml.use('Agg')
import matplotlib.pyplot as plt


def partitionData():
    X_train = np.genfromtxt("./../Feature Engineering/train_final.csv", delimiter=",", dtype=float)
    X_test = np.genfromtxt("./../Feature Engineering/test_final.csv", delimiter=",", dtype=float)
    return X_train[1:, :-1], X_train[1:, -1], X_test[1:, :-1], X_test[1:, -1]  # X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = partitionData()


print(X_train.shape)




print(X_train.shape)
# cut = int (np.shape(X_train)[0] * 0.8)
# X_Validation = X_train[cut:,:]
# X_Train = X_train[:cut,:]
# Y_Validation = Y_train[cut:]
# Y_Train = Y_train[:cut]

train_acc_l1 = []
validation_acc_l1 = []
test_acc_l1 = []

train_acc_l2 = []
validation_acc_l2 = []
test_acc_l2 = []


for i in range(10,51):
    pca = PCA(n_components=50)
    x_transformed = pca.fit_transform(X_train)
    xt_transformed = pca.transform(X_test)

    cut = int (np.shape(x_transformed)[0] * 0.8)
    X_Validation = x_transformed[cut:,:]
    X_Train = x_transformed[:cut,:]
    Y_Validation = Y_train[cut:]
    Y_Train = Y_train[:cut]

    clf = LogisticRegression(solver='saga', penalty='l1').fit(X_Train, Y_Train)
    l1_train = clf.score(X_Train, Y_Train)
    l1_valid = clf.score(X_Validation, Y_Validation)
    l1_test = clf.score(xt_transformed, Y_test)
    print("i = {} L1 -- train_acc = {} valid_acc = {} test_acc = {}".format(i, l1_train, l1_valid, l1_test))
    train_acc_l1.append(l1_train)
    validation_acc_l1.append(l1_valid)
    test_acc_l1.append(l1_test)

    clf = LogisticRegression(solver='saga', penalty='l2').fit(X_Train, Y_Train)
    l2_train = clf.score(X_Train, Y_Train)
    l2_valid = clf.score(X_Validation, Y_Validation)
    l2_test = clf.score(xt_transformed, Y_test)
    print("i = {} L2 -- train_acc = {} valid_acc = {} test_acc = {}".format(i, l2_train, l2_valid, l2_test))
    train_acc_l2.append(l2_train)
    validation_acc_l2.append(l2_valid)
    test_acc_l2.append(l2_test)


plt.plot(range(10,51), train_acc_l1, label = 'L1 Train')
plt.plot(range(10,51), train_acc_l2, label = 'L2 Train')
plt.plot(range(10,51), test_acc_l1, label = 'L1 Test')
plt.plot(range(10,51), test_acc_l2, label = 'L2 Test')
plt.plot(range(10,51), validation_acc_l1, label = 'L1 Valid')
plt.plot(range(10,51), validation_acc_l2, label = 'L2 Valid')
plt.ylabel("Accuracy")
plt.xlabel("PCA Dimension Kept")
plt.title('Accuracy of Train, Test and Validation on L1 and L2 Pen')
plt.legend()
plt.savefig("./overall.png")

print(train_acc_l1)
print("L1")
print(validation_acc_l1)
print("L1")
print(test_acc_l1)
print("##########################")
print(train_acc_l2)
print("L2")
print(validation_acc_l2)
print("L2")
print(test_acc_l2)


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
# print("Training Start")
# logist = LogisticRegressionCV(cv=2, solver='saga', max_iter=10000, penalty='l1')
# logist.fit(X_train, Y_train)
# print(logist.score(X_test, Y_test))

# plot in terms of cv
# plot in terms of penalty
# plot in terms of
cv = range(2,6)
solvers = ['sag','saga','lbfgs']
penalty = ['l1','l2']
# for i in range(50):


# logisc_t = {}
# logisc_v = {}
#
# for i in tqdm(cv):
#     print("Training with CV = {}".format(i))
#     for pen in tqdm(penalty):
#         print("Penalty = {}".format(pen))
#         logist = LogisticRegressionCV(cv=i, solver='saga', max_iter=10000, penalty=pen)
#         logist.fit(X_train, Y_train)
#         key = "Pen={},CV={}".format(pen,i)
#         logisc_t[key] = logist.score(X_test, Y_test)
#         logisc_v[key] = logist.score(X_Validation, Y_Validation)
#
# plt.bar(list(logisc_t.keys()), list(logisc_t.values()))
# plt.title("Accuracy Test")
# plt.xlabel("Penalty with CV")
# plt.ylabel("Accuracy")
# plt.savefig("./LogisticTest.png")
# plt.close()
#
# plt.bar(list(logisc_v.keys()), list(logisc_v.values()))
# plt.title("Accuracy Validation")
# plt.xlabel("Penalty with CV")
# plt.ylabel("Accuracy")
# plt.savefig("./LogisticValidation.png")
# plt.close()
#
#
#
#
#
# print(logist.score(X_test, Y_test))
# print(logist)
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
