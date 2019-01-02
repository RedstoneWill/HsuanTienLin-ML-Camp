# homework 1

#Load the dataset

import numpy as np

def loadfile(file):
    X = [] # features, shape = (samples, features)
    Y = [] # labels, shape = (sample,)
    for line in open(file).readlines():
        items = line.strip().split('\t') # features and label split by Tab
        y = items[1].strip()
        y = float(y) # str to float
        Y.append(y)
        x = items[0].strip().split(' ')
        x = list(map(float, x)) # str to float
        X.append(x)
    X = np.array(X) # list to array
    Y = np.array(Y) # list to array
    return X, Y