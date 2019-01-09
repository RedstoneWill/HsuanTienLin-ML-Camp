import numpy as np

class RidReg(object):

    def __init__(self):
        pass
    

    # Q1: if lambda = 11.26, what is Ein and Eout?
    # define load the data function
    def read_input_data(self, path):
        x = []
        y = []
        for line in open(path).readlines():
            items = line.strip().split(' ')
            tmp_x = []
            for i in range(0, len(items) - 1): 
                tmp_x.append(float(items[i]))
            x.append(tmp_x)
            y.append(float(items[-1]))
        return np.array(x), np.array(y)

    # define ridge regression for classification by closed form solution
    def ridge_regression(self, X, y, lamda = 0):
        # ridge regression closed form solution
        ### YOUR CODE HERE



        ### END YOUR CODE
    
        return W
    
    # define calculate error function
    def cal_error(self, X, y, W):
        
        ### YOUR CODE HERE



        ### END YOUR CODE
    
        return error
    
    # define calculate Ein and Eout
    def cal_Ein_Eout(self, X_train, y_train, X_test, y_test, lamda=11.26):
        
        ### YOUR CODE HERE



        ### END YOUR CODE

        return Ein, Eout
    

    # Q2: train data & validation data.calculate Eval & Eout respect different lambda
    def cal_val(self, X_train, y_train, X_test, y_test):

        # split train data to train and val
        X_Dtrain = X_train[:120]  # first 120 samples
        y_Dtrain = y_train[:120]
        X_Dval = X_train[-80:]  # last 80 samples
        y_Dval = y_train[-80:]

        lamda_log = [i for i in range(2, -11, -1)]  # log10(lamda)
        Eval_best = 1.0  # min Eval
        W_best = np.zeros(X_train.shape[1]).reshape(-1,1)  # W of min Eval
        lamda_best = 0  # initialze lambda 
        Eval_all = []  # store all Eval

        
        ### YOUR CODE HERE



        ### END YOUR CODE

        return lamda_best, Eval_best, Eout
    
    # Q3: 5-folds cross validation
    # define 5-folds cross validation
    def cross_val(self, X, y):
        lamda_log = [i for i in range(2, -11, -1)]  # log10(lamda)
        Eval_best = 1.0  # min Eval
        W_best = np.zeros(X.shape[1]).reshape(-1,1)  # W of min Eval
        lamda_best = 0  # initialze lambda 
        Eval_all = []  # store all Eval in different lambda
    
        k = 5  # k flod cross-validation
        num_flod = int(X.shape[0] / k)  # samples  of one floder
        
        ### YOUR CODE HERE



        ### END YOUR CODE
                
        return lamda_best, Eval_best, Eval_all







