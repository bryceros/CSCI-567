import numpy as np
from typing import List


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        # ∑n=1NMAX(0,−yn(wTxn+b))
        y_bin = 2.0*y-1.0
        for iteration in range(max_iterations):
            sign = np.array([(lambda z: 1.0 if z <= 0.0 else 0.0)(z) for z in y_bin*(np.matmul(w,X.T)+b)])
            b_grad = np.matmul(sign,y_bin.T)
            w_grad = np.matmul(sign*y_bin,X)
            w += (step_size/N)*w_grad
            b += (step_size/N)*b_grad
        ############################################
  
    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        for iteration in range(max_iterations):
            w_grad,b_grad = np.zeros(D),0
            error = sigmoid(np.dot(w,X.T) + b) - y
            w_grad += np.dot(error,X)
            b_grad += np.sum(error)
            w -= (step_size/N) * w_grad
            b -= (step_size/N) * b_grad 
        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value
    value =  1/(1+np.exp(-1*z))
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.array([(lambda X,w,b: 1.0 if np.matmul(w,X.T)+b > 0.0 else 0)(X[n],w,b) for n in range(N)])
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        preds = np.array([(lambda X,w,b: 1.0 if sigmoid(np.matmul(w,X.T)+b) > 0.5 else 0.0)(X[n],w,b) for n in range(N)])
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    def softmax(x):
        if x.ndim == 1:
            x = np.exp(x - np.amax(x)).reshape(-1,1)
            return x.T / np.sum(x)
        else:
            x = np.exp(x - np.amax(x,axis=1).reshape(-1,1))
            return x.T / np.sum(x,axis=1)


    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        y_one_hot = np.eye(C)[y]
        y_one_hot = y_one_hot.T
        X = np.append(X,np.ones((N,1)),axis=1)  
        X_T = X.T
        w = np.column_stack((w,b))
        for iteration in range(max_iterations):
            rand = np.random.choice(N)
            z = np.dot(w,X_T[:,rand]).T
            soft_max = softmax(z) - y_one_hot[:,rand]
            soft_max_error = np.dot(soft_max.reshape(-1,1),X[rand,:].reshape(1,-1))
            w -= (step_size)*soft_max_error
        b = w[:,-1]
        w = w[:,:-1]
        #w = np.zeros((C, D))
        #b = np.zeros(C)
        ############################################
    

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        #x_n = get_branches(features=X,labels=y,C=C)
        
        #X = np.append(X,np.ones((N,1)),axis=1) 
        #w = np.column_stack((w,b))
        
        y_one_hot = np.eye(C)[y]
        y_one_hot = y_one_hot.T
        X = np.append(np.ones((N,1)),X,axis=1)  
        X_T = X.T
        w = np.column_stack((b,w))
        for iteration in range(max_iterations):
            z = np.dot(w,X_T).T
            soft_max = softmax(z) - y_one_hot
            soft_max_error = np.dot(soft_max,X)
            w -= (step_size/N)*soft_max_error
        b = w[:,0]
        w = w[:,1:]

            
            
        #w = np.zeros((C, D))
        #b = np.zeros(C)
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    def softmax(x):
        if x.ndim == 1:
            x = np.exp(x - np.amax(x)).reshape(-1,1)
            return x.T / np.sum(x)
        else:
            x = np.exp(x - np.amax(x,axis=1).reshape(-1,1))
            return x.T / np.sum(x,axis=1)

    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    X = np.append(np.ones((N,1)),X,axis=1)  
    X_T = X.T
    w = np.column_stack((b,w))
    soft_max = softmax(np.dot(w,X_T).T).T
    preds = np.argmax(soft_max,axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds




        