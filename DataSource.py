import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder

# def process_data():
#     X = np.zeros((1,6))
#     Y = np.zeros((1,1))
#     counter = 0
#     for chunk_df in pd.read_csv('train.csv', chunksize = 1000):
#         chunk_ma = chunk_df.as_matrix()
#         chunk_ma = chunk_ma[np.argsort(chunk_ma[:, 7])]
#         i = 999
#         while chunk_ma[i][7] == 1:
#             Y = np.vstack([Y,[1]])
#             X = np.vstack([X, chunk_ma[i][:6]])
#             i = i-1
            
#         temp = random.sample(range(i), 2)
        
#         Y = np.vstack([Y,[0]])
#         X = np.vstack([X, chunk_ma[random.randint(temp[0], i)][:6]])
        
#         Y = np.vstack([Y,[0]])
#         X = np.vstack([X, chunk_ma[random.randint(temp[1], i)][:6]])
#         counter = counter + 1
#         print(counter)
    
#     df = pd.DataFrame(X[1:])
#     df.to_csv("X.csv")
#     df = pd.DataFrame(Y[1:])
#     df.to_csv("Y.csv")

# Use Sklearn to do one hot encoding
# le_color = LabelEncoder()
# le_make = LabelEncoder()
# df['color_encoded'] = le_color.fit_transform(df.color)
# df['make_encoded'] = le_make.fit_transform(df.make)

def partitionData():
        X = np.genfromtxt("output_data.csv", delimiter=",")
        X = X[1:,:] # Excluse first row which is title names
        train_C = int(X.shape[0] * 0.8)
        train = X[:train_C,:]
        test = X[(train_C + 1):,:]
        # print(train[-1])
        # print(test[1])
        print(test[:,-1])
        print(np.shape(train[:,:-1]))
        print(np.shape(train[:,-1]))
        print(np.shape(test[:, :-1]))
        print(np.shape( test[:,-1]))
        return train[:,:-1], train[:,-1], test[:, :-1], test[:,-1]
partitionData()
# def run():
#         df =  pd.read_csv('dataset.csv', delimiter=',')
#         app_ohe = OneHotEncoder()
#         device_ohe = OneHotEncoder()
#         os_ohe = OneHotEncoder()
#         channel_ohe = OneHotEncoder()

#         app_X = app_ohe.fit_transform(df.app.values.reshape(-1,1)).toarray()
#         device_ohe_X = device_ohe.fit_transform(df.device.values.reshape(-1,1)).toarray()
#         os_ohe_X = os_ohe.fit_transform(df.os.values.reshape(-1,1)).toarray()
#         channel_ohe_X = channel_ohe.fit_transform(df.channel.values.reshape(-1,1)).toarray()

#         dfOneHot = pd.DataFrame(app_X, columns = ["app_" + str(int(i)) for i in range(app_X.shape[1])])
#         df = pd.concat([df, dfOneHot], axis = 1)

#         dfOneHot = pd.DataFrame(device_ohe_X, columns=["device_"+str(int(i)) for i in range (device_ohe_X.shape[1])])
#         df = pd.concat([df, dfOneHot], axis = 1)

#         dfOneHot = pd.DataFrame(os_ohe_X, columns=["os_" + str(int(i)) for i in range(os_ohe_X.shape[1])])
#         df = pd.concat([df, dfOneHot], axis = 1)

#         dfOneHot = pd.DataFrame(channel_ohe_X, columns=["channel_" + str(int(i)) for i in range(channel_ohe_X.shape[1])])
#         df = pd.concat([df, dfOneHot], axis = 1)
        
#         columns = ['app', 'device', 'os', 'channel']
#         df.drop(columns, inplace=True, axis=1)
#         df.to_csv('output_data.csv')
        
# run()

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


# dfOneHot = pd.DataFrame(X, columns = ["Color_"+str(int(i)) for i in range(X.shape[1])])
# df = pd.concat([df, dfOneHot], axis=1)

# dfOneHot = pd.DataFrame(Xm, columns = ["Make"+str(int(i)) for i in range(Xm.shape[1])])
# df = pd.concat([df, dfOneHot], axis=1)

# dfOneHot = pd.DataFrame(X, columns = ["Color_"+str(int(i)) for i in range(X.shape[1])])
# df = pd.concat([df, dfOneHot], axis=1)
# dfOneHot = pd.DataFrame(Xm, columns = ["Make"+str(int(i)) for i in range(X.shape[1])])
# df = pd.concat([df, dfOneHot], axis=1)




