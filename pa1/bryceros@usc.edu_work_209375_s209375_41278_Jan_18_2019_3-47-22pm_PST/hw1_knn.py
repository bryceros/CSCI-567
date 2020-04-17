from __future__ import division, print_function

from typing import List, Callable

import numpy as np
import scipy
from collections import Counter


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function, scaling_class=None):
        self.k = k
        self.distance_function = distance_function
        self.scaling_class = scaling_class
        self.training_example_labels = List[int]
        self.training_example_features = List[List[float]]

    #TODO: Complete the training function
    def train(self, features: List[List[float]], labels: List[int]):
        self.training_example_features = features
        self.training_example_labels = labels
        
    #TODO: Complete the prediction function
    def predict(self, features: List[List[float]]) -> List[int]:
        labels = []
        for unlabel_feature in features:
            temp = self.get_k_neighbors(unlabel_feature)
            label = Counter(temp).most_common(1)[0][0]
            labels.append(label)
        return labels

    #TODO: Complete the get k nearest neighbor function
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        #raise NotImplementedError
        '''ret_distance = []
        ret_label = []
        for feature,label in zip(self.training_example_features,self.training_example_labels):
            distance = self.distance_function(feature,point)
            if len(ret_distance) < self.k and len(ret_label) < self.k:
                ret_distance.append(distance)
                ret_label.append(label)
            else:
                max_distance = max(ret_distance)
                if distance < max_distance: 
                    max_index = ret_distance.index(max_distance)
                    ret_distance[max_index] = distance
                    ret_label[max_index] = label
        return ret_label'''
        ret_list = []
        for feature,label in zip(self.training_example_features,self.training_example_labels):
            ret_list.append([self.distance_function(feature, point),label])
        ret_list.sort(key=lambda x: x[0])
        return (np.array(ret_list[0:self.k])[:,1]).tolist()

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
