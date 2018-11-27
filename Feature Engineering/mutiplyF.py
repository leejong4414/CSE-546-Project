import numpy as np

def mutiplyLine(Matrix, l1, l2):
    l3 = np.multiply(Matrix[:,l1], Matrix[:,l2])
    Matrix = np.concatenate((Matrix, np.array([l3]).T),axis=1)
    return Matrix