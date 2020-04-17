from typing import List, Callable, Dict

import numpy as np
import matplotlib.pyplot as plt
from hw1_knn import KNN
from hw1_dt import DecisionTree, TreeNode

#TODO: Information Gain function
def Information_Gain(S: float, branches: List[List[any]]) -> float:
    return S - Conditional_Entropy(branches)

def Conditional_Entropy(branches: List[List[any]]):
    #(P(A = v) * H(Y | A = v)
    total_branch_class_count = [ np.sum(branch_class_division) for branch_class_division in branches]
    total_class_count = np.sum(total_branch_class_count)
    return np.sum([branch_class_count/total_class_count * Weighted_Average_Entropy(branch_class_division) for branch_class_division,branch_class_count, in zip(branches,total_branch_class_count)])

def Weighted_Average_Entropy(class_division: List[any]) -> float:
    # sum P(Y = k)logP(Y = k)
    total_labels = np.sum(class_division)
    return np.sum([ (lambda class_num,total_labels: -1.0*(float(class_num)/float(total_labels)) * np.log2(float(class_num)/float(total_labels)) if class_num != 0 else 0)(class_num,total_labels) for class_num in class_division])
    
# TODO: implement reduced error pruning
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List[any]
    reduced_error_prunning_helper(decisionTree.root_node,X_test,y_test)
    #reduced_error_prunning_helper2(decisionTree.root_node,decisionTree.predict,X_test,y_test)
    
def reduced_error_prunning_helper(node: TreeNode, X_test: List[List[any]], y_test: List[int]) -> bool:
    assert(len(X_test) == len(y_test))
    if node.children == [] or node.splittable == False:
        return True
    elif len(X_test) == 0 and len(y_test) == 0:
        node.children = []
        node.splittable = False
        return True
    # get branches from X_test y_test set
    branches = get_branches(X_test,y_test,node.dim_split)

    if any([reduced_error_prunning_helper(node.children[node.feature_uniq_split.index(v)],list(branches[v])[0],list(branches[v])[1]) for v in list(branches.keys())]):
        pred_no_pruning = get_accuracy([ node.predict(feature) for feature in X_test],y_test)
        pred_pruning = get_accuracy([ node.cls_max for feature in X_test],y_test)
        if pred_pruning >= pred_no_pruning:
            node.children = []
            node.splittable = False
            return True
    return False

    
'''def reduced_error_prunning_helper2(node: TreeNode,pred,f,l) -> bool:
    if node.children == []:
        return True

    if any([reduced_error_prunning_helper2(child,pred,f,l) for child in node.children]):
        pred_no_pruning = get_accuracy(pred(f),l)
        childs = node.children
        node.children = []
        node.splittable = False
        pred_pruning = get_accuracy(pred(f),l)
        node.children = childs
        node.splittable = True
        if pred_pruning >= pred_no_pruning:
            node.children = []
            node.splittable = False
            return True
    return False'''

# print current tree
# Do not change this function
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')

def get_amount_cls(labels: List[int])->List[int]:
    # this returns a array of the number of time a given class occours
    class_count = {} 
    for label in labels:
        if label in class_count.keys():
            class_count[label] += 1
        else:
            class_count[label] = 1
    return list(class_count.values())

def remove_and_return(array: List[any], index: int) ->List[any]:
    array_copy = list(array[:])
    del array_copy[index]
    return array_copy

def get_branches(features: List[List[any]], labels: List[int], A: int): #-> Dict[any , List[ (List[List[any]], List[int]) ]]:
    # this give a dict key: v in A and a value of both (features, labels) that have the feature v
    ret_val = {}
    for label,feature in zip(labels, features):
        if feature[A] in ret_val:
            ret_val[feature[A]][0].append(remove_and_return(feature,A))
            ret_val[feature[A]][1].append(label)
        else:
            ret_val[feature[A]] = [[remove_and_return(feature,A)],[label]]
    return ret_val

def get_accuracy(labels1: List[int],labels2: List[int]) -> float:
    assert(len(labels1) == len(labels2))
    correct = 0
    for i, j in zip(labels1, labels2):
        if i == j:
            correct +=1
    return float(correct)/float(len(labels1))
#KNN Utils

#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    """
    f1 score: https://en.wikipedia.org/wiki/F1_score
    """
    assert len(real_labels) == len(predicted_labels)
    #raise NotImplementedError
    return(2.0*inner_product_distance(real_labels,predicted_labels))/(np.sum(real_labels)+np.sum(predicted_labels))
    
#TODO: Euclidean distance, inner product distance, gaussian kernel distance and cosine similarity distance

def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    return np.sqrt(np.sum([np.float_power(p1-p2,2) for p1,p2 in zip(point1,point2)]))

def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    return np.dot(point1, point2)
    #return np.sum([p1*p2 for p1,p2 in zip(point1,point2)])

def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    return -1*np.exp(-0.5*np.sum([np.float_power(p1-p2,2) for p1,p2 in zip(point1,point2)]))

def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    #raise NotImplementedError
    return 1.0 - (inner_product_distance(point1,point2) \
    /(np.sqrt(inner_product_distance(point1,point1)*inner_product_distance(point2,point2))))

#TODO: Complete the model selection function where you need to find the best k     
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
        func_names = list(distance_funcs.keys())
        func_names.reverse()
        best_model = KNN(k=np.inf,distance_function=None)
        best_name = func_names[0]
        best_valid_f1_score = -1*np.inf
        for k in range(1, min(30,len(Xtrain)-1), 2):
            for item in distance_funcs.items():
                name,distance_func = item
                model = KNN(k=k,distance_function=distance_func)
                model.train(Xtrain,ytrain)
                train_predict_labels = model.predict(Xtrain)
                train_f1_score = f1_score(ytrain,train_predict_labels)
                valid_predict_labels = model.predict(Xval)
                valid_f1_score = f1_score(yval,valid_predict_labels)

                if valid_f1_score > best_valid_f1_score or \
                (valid_f1_score == best_valid_f1_score and func_names.index(name) > func_names.index(best_name)):
                    best_model = model
                    best_name = name
                    best_valid_f1_score = valid_f1_score

                #Dont change any print statement
                '''print('[part 1.1] {name}\tk: {k:d}\t'.format(name=name, k=model.k) + 
                        'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                        'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
                print()
                print('[part 1.1] {name}\tbest_k: {best_k:d}\t'.format(name=best_name, best_k=best_model.k) +
                    'valid f1 score: {valid_f1_score:.5f}'.format(valid_f1_score=best_valid_f1_score))
                print()'''
        return best_model, best_model.k, best_name

#TODO: Complete the model selection function where you need to find the best k with transformation
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):    
    s_dict = {}
    for s_item in scaling_classes.items():
        s_name,s_func = s_item
        scaling = s_func()
        scale_train = scaling(Xtrain)
        scale_valid = scaling(Xval)
        s_dict[s_name] = [scale_train,scale_valid]
    func_names = list(distance_funcs.keys())
    func_names.reverse()
    scaling_name = list(scaling_classes.keys())
    scaling_name.reverse()
    best_model = KNN(k=np.inf,distance_function=None, scaling_class=None)
    best_name = func_names[0]
    best_scaling_name = scaling_name[0]
    best_valid_f1_score = -1*np.inf
    for k in range(1, min(30,len(Xtrain)-1), 2):
        for item in distance_funcs.items():
            for s_name in scaling_name:
                name,distance_func = item
                model = KNN(k=k,distance_function=distance_func, scaling_class=scaling_classes[s_name])
                model.train(s_dict[s_name][0],ytrain)
                train_predict_labels = model.predict(s_dict[s_name][0])
                train_f1_score = f1_score(ytrain,train_predict_labels)
                valid_predict_labels = model.predict(s_dict[s_name][1])
                valid_f1_score = f1_score(yval,valid_predict_labels)

                if valid_f1_score > best_valid_f1_score or \
                (valid_f1_score == best_valid_f1_score and scaling_name.index(s_name) > scaling_name.index(best_scaling_name)) or \
                (valid_f1_score == best_valid_f1_score and scaling_name.index(s_name) == scaling_name.index(best_scaling_name) and func_names.index(name) > func_names.index(best_name)):
                    best_model = model
                    best_name = name
                    best_scaling_name = s_name
                    best_valid_f1_score = valid_f1_score

                #Dont change any print statement
                '''print('[part 1.1] {name}\t{s_name}\tk: {k:d}\t'.format(name=name, s_name=s_name, k=model.k) + 
                        'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                        'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
                print()
                print('[part 1.1] {name}\t{s_name}\tbest_k: {best_k:d}\t'.format(name=best_name, s_name=best_scaling_name, best_k=best_model.k) +
                    'valid f1 score: {valid_f1_score:.5f}'.format(valid_f1_score=best_valid_f1_score))
                print()'''
    return best_model, best_model.k, best_name, best_scaling_name

def classify(model,Xtest,ytest):
    if model.scaling_class is not None:
        scaling = model.scaling_class()
        Xtest = scaling(Xtest)
    predict_labels = model.predict(Xtest)
    return f1_score(ytest,predict_labels)

class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        #raise NotImplementedError
        ret_val = []
        for feature in features:
            if np.count_nonzero(feature) :
                unit = np.sqrt(inner_product_distance(feature,feature))
                ret_val.append([x / unit for x in feature])
            else:
                ret_val.append(np.zeros_like(feature).tolist())

        return ret_val

class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.
    Note:
        1. you may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).
    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]
        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]
        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]
        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """

    def __init__(self):
        self.min = None
        self.max = None

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        #raise NotImplementedError
        ret_val = []
        if self.max is None and self.min is None:
            self.min = np.amin(features,axis=0)
            self.max = np.amax(features,axis=0)
        for feature in features:
            temp_list = []
            for x,min,max in zip(feature,self.min,self.max):
                if max - min != 0:
                    temp_list.append((x - min) / (max - min))
                else:
                    temp_list.append(x - min)
            ret_val.append(temp_list)
        return ret_val        
        #return [ [(x - min) / (max-min) for x,min,max in zip(feature,self.min,self.max)] for feature in features]