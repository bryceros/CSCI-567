from utils import euclidean_distance, gaussian_kernel_distance, inner_product_distance, cosine_sim_distance, NormalizationScaler, MinMaxScaler, f1_score
import numpy as np

print('hi: ', cosine_sim_distance([3, 45, 7, 2],[2, 54, 13, 15]))
x = [-1,2,3]
y = [4,0,-3]
assert(euclidean_distance(x,y) == np.sqrt(65))
assert(gaussian_kernel_distance(x,y) == -1*np.exp(-32.5))
assert(cosine_sim_distance([3, 45, 7, 2],[2, 54, 13, 15]) == 0.972284251712)
normal = NormalizationScaler()
assert(normal([[3, 4], [1, -1], [0, 0]]) == [[0.6, 0.8], [0.7071067811865475, -0.7071067811865475], [0, 0]])
minMax = MinMaxScaler()
assert(minMax([[0, 10], [2, 0]]) == [[0, 1], [1, 0]])
assert(minMax([[20, 1]]) == [[10, 0.1]])
newMinMax = MinMaxScaler()
assert(newMinMax([[1, 1], [0, 0]]) == [[1.0, 1.0], [0.0, 0.0]])
assert(newMinMax([[20, 1]]) == [[20, 1]])
assert(f1_score([1,1,1,1], [1,0,0,0]) == 0.4)

import numpy as np
from hw1_knn import KNN
from utils import euclidean_distance, gaussian_kernel_distance, inner_product_distance, cosine_sim_distance
from utils import classify, f1_score, model_selection_without_normalization, model_selection_with_transformation
distance_funcs = {
    'euclidean': euclidean_distance,
    'gaussian': gaussian_kernel_distance,
    'inner_prod': inner_product_distance,
    'cosine_dist': cosine_sim_distance,
}
scaling_classes = {
    'min_max_scale': MinMaxScaler,
    'normalize': NormalizationScaler,
}
from data import data_processing
Xtrain = [[1,1],[1,1.25],[1.25,1],[1.4,1.75],   [1.75,1.75],[1.80,1.75],[2,1.75],[1.75,2.25],[2,2.5],     [2,3],[2.15,3],[2.45,3],[2.5,3],[2.75,3],[3,3]]
ytrain = [ [0],[0],[0],[0], [1],[1],[1],[1],[1], [2],[2],[2],[2],[2],[2]]
Xtest = data_processing()

best_model, best_k, best_function = model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xtrain, ytrain)
print(pr(best_model,Xtest,ytest))

'''
from utils import Information_Gain, get_branches ,get_amount_cls
features =  [[2,     3,      4,    15,     20],
           [4,     6,      11,   13,     15],
           [2,     5,      10,   13,     20],
           [7,     10,     11,   18,     19],
           [11,    13,     16,   18,     19],
           [3,     5,      11,   15,     18],
           [3,     6,      8,    12,     15],
           [5,     11,     15,   17,     18],
           [2,     3,      4,    12,     18],
           [2,     4,      9,    12,     15]]
labels = [19,10,16,13,18,10,3,8,10,17]

assert(Information_Gain(S = 0.97, branches = [[2, 5], [10, 3]]) == 0.16133040676182386)
'''
'''import data
import hw1_dt as decision_tree
import utils as Utils
from sklearn.metrics import accuracy_score

#load data
X_train, y_train, X_test, y_test = data.load_decision_tree_data()

# set classifier
dTree = decision_tree.DecisionTree()

# training
dTree.train(X_train.tolist(), y_train.tolist())

import json
# testing
y_est_test = dTree.predict(X_test)
test_accu = accuracy_score(y_est_test, y_test)
print('test_accu', test_accu)

Utils.reduced_error_prunning(dTree, X_test, y_test)

# print
Utils.print_tree(dTree)

y_est_test = dTree.predict(X_test)
test_accu = accuracy_score(y_est_test, y_test)
print('test_accu', test_accu) '''