import numpy as np
import pandas as pd


X = pd.read_csv("train_final_OHE.csv")


X = X.as_matrix()
print(np.shape(X))
print(X[0])
print(X[1])
print(X[2])
X_new = np.append(X[:,1],X[:,3:], axis=1)
X_new = np.append(X_new,X[:,2], axis=1)
# rand_idx = np.random.choice(x, (n), replace=False)
# X = X[rand_idx]
np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
end = int(X.shape[0]*0.9)
train_X = X[:end,:]
test_X = X[end:,:]
print(X_new[0])
print(X_new[1])
print(X_new[2])
print(np.shape(train_X))
print(np.shape(test_X))


# pd.DataFrame(train_X).to_csv('train_final.csv')
# pd.DataFrame(test_X).to_csv('test_final.csv')