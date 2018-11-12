import numpy as np
import pandas as pd
import random

def process_data():
    X = np.zeros((1,6))
    Y = np.zeros((1,1))
    counter = 0
    for chunk_df in pd.read_csv('train.csv', chunksize = 1000):
        
        chunk_ma = chunk_df.as_matrix()
        chunk_ma = chunk_ma[np.argsort(chunk_ma[:, 7])]
        i = 999
        while chunk_ma[i][7] == 1:
            Y = np.vstack([Y,[1]])
            X = np.vstack([X, chunk_ma[i][:6]])
            i = i-1
            
        temp = random.sample(range(i), 2)
        
        Y = np.vstack([Y,[0]])
        X = np.vstack([X, chunk_ma[random.randint(temp[0], i)][:6]])
        
        Y = np.vstack([Y,[0]])
        X = np.vstack([X, chunk_ma[random.randint(temp[1], i)][:6]])
        counter = counter + 1
        print(counter)
    
    df = pd.DataFrame(X[1:])
    df.to_csv("X.csv")
    df = pd.DataFrame(Y[1:])
    df.to_csv("Y.csv")

process_data()

#==========================================================================================#
#==========================================================================================#

# Use Sklearn to do one hot encoding
# 

# le_color = LabelEncoder()
# le_make = LabelEncoder()
# df['color_encoded'] = .fit_transform(df.color)
# df['make_encoded'] = le_make.fit_transform(df.make)
#

# from sklearn.preprocessing import OneHotEncoder

# df = pd.DataFrame([
#        [0, 1, 2017],
#        [1, 0, 2015], 
#        [2, 1, 2018],
# ])
# df.columns = ['color', 'make', 'year']

# color_ohe = OneHotEncoder()
# make_ohe = OneHotEncoder()
# X = color_ohe.fit_transform(df.color.values.reshape(-1,1)).toarray()
# Xm = make_ohe.fit_transform(df.make.values.reshape(-1,1)).toarray()

# print(Xm)

# dfOneHot = pd.DataFrame(X, columns = ["Color_"+str(int(i)) for i in range(X.shape[1])])
# df = pd.concat([df, dfOneHot], axis=1)

# dfOneHot = pd.DataFrame(Xm, columns = ["Make"+str(int(i)) for i in range(Xm.shape[1])])
# df = pd.concat([df, dfOneHot], axis=1)

# dfOneHot = pd.DataFrame(X, columns = ["Color_"+str(int(i)) for i in range(X.shape[1])])
# df = pd.concat([df, dfOneHot], axis=1)
# dfOneHot = pd.DataFrame(Xm, columns = ["Make"+str(int(i)) for i in range(X.shape[1])])
# df = pd.concat([df, dfOneHot], axis=1)


#==========================================================================================#
#==========================================================================================#




