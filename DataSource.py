import numpy as np
import pandas as pd
# Load a csv of floats:
# Load a text file of integers:
# y = np.loadtxt("upvote_labels.txt", dtype=np.int)
# Load a text file of strings:
# featureNames = open("upvote_features.txt").read().splitlines()
for chunk_df in pd.read_csv('train.csv', chunksize = 100):
    print(chunk_df.values)
