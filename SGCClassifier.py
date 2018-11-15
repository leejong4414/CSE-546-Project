from sklearn.linear_model import SGDClassifier
import DataSource as ds


train_x, train_y, test_x, test_y = ds.partitionData()

clf2 = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001)
clf2.fit(train_x, train_y)

print(clf2.score(test_x, test_y))
print(clf2.score(train_x, train_y))


clf3 = SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.0001)
clf3.fit(train_x, train_y)

print(clf3.score(test_x, test_y))
print(clf3.score(train_x, train_y))


