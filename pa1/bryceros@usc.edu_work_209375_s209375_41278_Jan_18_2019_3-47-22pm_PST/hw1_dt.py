from typing import List, Callable
import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features: List[List[float]], labels: List[int]): 
        # features: List[List[float]], labels: List[int]
        # init
        if type(features) is np.ndarray:
            features = list(features)
        if type(labels) is np.ndarray:
            labels = list(labels)

        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size
        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
        
        return
    

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred
        
class TreeNode(object):
    def __init__(self, features: List[List[any]], labels: List[int], num_cls: int):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
        # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2 or any( i == [] for i in features ):
            self.splittable = False
        else:
            self.splittable = True
    

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = []  # the possible unique values of the feature to be split

        self.debug_path = [0]

    # TODO: implement split function
    def split(self):
        s = Util.Weighted_Average_Entropy(Util.get_amount_cls(self.labels))
        best_gain_value = -1.0*np.inf
        best_branches = {}
        best_a = 0
        best_class_amounts = []
        # find best attributes to spilt
        for a in range(len(self.features[0])):
            branches = Util.get_branches(self.features, self.labels,a)
            class_amounts = [ Util.get_amount_cls(branch[1]) for branch in list(branches.values())]
            gain_value = Util.Information_Gain(s, class_amounts)
            if (gain_value > best_gain_value) or (gain_value == best_gain_value and len(branches.keys()) > len(best_branches.keys())):
                best_a = a
                best_class_amounts = class_amounts
                best_gain_value = gain_value
                best_branches = branches
        # setup the selected attributes splits
        self.dim_split = best_a
        self.feature_uniq_split = list(best_branches.keys())
       
        features = [data[0] for data in list(best_branches.values())]
        labels = [data[1] for data in list(best_branches.values())]
               
        best_class_amounts = [x for _,x in sorted(zip(self.feature_uniq_split,best_class_amounts),reverse = False)]
        features = [feature for _,feature in sorted(zip(self.feature_uniq_split,features),reverse = False)]
        labels = [label for _,label in sorted(zip(self.feature_uniq_split,labels),reverse = False)]
        self.feature_uniq_split = sorted(self.feature_uniq_split)

        self.children = [ TreeNode(feature,label,len(class_amount)) for feature,label,class_amount in zip(features,labels,best_class_amounts)]
        #self.children = [ TreeNode(data[0],data[1],len(class_amount)) for data,class_amount in zip(list(best_branches.values()),best_class_amounts)]
        debug_i = 0
        for child in self.children:
            child.debug_path = self.debug_path + [debug_i]
            debug_i +=1
            if child.splittable:
               child.split()
        
    # TODO:treeNode predict function
    def predict(self, feature: List[any]) -> int:
        if self.children == [] or feature[self.dim_split] not in self.feature_uniq_split:
            return self.cls_max
        else:
            return self.children[self.feature_uniq_split.index(feature[self.dim_split])].predict(Util.remove_and_return(feature,self.dim_split))
    