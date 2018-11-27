import numpy as np
import pandas as pd

#/Users/Eric/Desktop/CSE-546-Project/Feature Engineering/train.csv
X = pd.read_csv("")

x = X.shape[0]
n = X.shape[0]

rand_idx = np.random.choice(x, (n), replace=False)
X = X[rand_idx]

end = int(x*0.9)
train_X = X[0:end,:]
test_X = X[end:,:]

pd.DataFrame(train_X).to_csv('train_final.csv')
pd.DataFrame(test_X).to_csv('test_final.csv')