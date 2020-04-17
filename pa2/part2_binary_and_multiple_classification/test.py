
import numpy as np
from typing import List

def get_weight_k(weight:List[List],C:int):
        ret_val = []
        for k in range(C):
            hold = np.delete(weight, k, axis=0)
            hold -= weight[k]
            ret_val.append(hold)
        return np.array(ret_val)

def get_branches(features: List[List[float]], labels: List[int],C:int): #-> Dict[any , List[ (List[List[any]], List[int]) ]]:
# this give a dict key: v in A and a value of both (features, labels) that have the feature v
    ret_val = [[] for i in range(C)]
    for label,feature in zip(labels, features):
        ret_val[label].append(feature)
    return np.array(ret_val)

step_size = 0.5
max_iterations = 1


X = np.array([[1.0,0.0,0.0],
              [0.0,1.0,0.0],
              [0.0,0.0,1.0],
              [0.0,0.0,1.0]])
y = np.array([0,1,2,2])
D,N = X.shape
C = np.unique(y).shape[0]

w = [[0.1,0.2,0.3],
     [0.1,0.2,0.3],
     [0.1,0.2,0.3]]
b = [0.01,0.1,0.1]

for iteration in range(max_iterations):
    x_not = get_branches(features=X,labels=y,C=C)
    wk_not = get_weight_k(w,C)
    softmax = []
    for k in range(C):
        temp = np.matmul(wk_not[k],X.T)+b
        loss = np.exp(temp-np.max(temp))
        z = np.sum(loss)
        # when k equal y
        softmax_k = 1.0+z/(1.0+z)
        # when k equal y
        for k_not in range(C-1):
            softmax_k += loss[k_not]/z
        softmax.append(softmax_k)
    softmax = np.array(softmax)

    w -= (step_size/N)*np.matmul(softmax,X)
    b -= (step_size/N)*np.sum(softmax)