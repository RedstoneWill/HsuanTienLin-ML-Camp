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
        W = np.linalg.inv(X.T.dot(X) + lamda * np.eye(X.shape[1])).dot(X.T).dot(y)
        W = W.reshape(-1,1)
    
        return W
    
    # define calculate error function
    def cal_error(self, X, y, W):
        y = y.reshape(-1, 1)
        scores = X.dot(W)
        y_pred = np.where(scores >= 0, 1, -1)
        error = np.mean(y_pred != y)
    
        return error
    
    # define calculate Ein and Eout
    def cal_Ein_Eout(self, X_train, y_train, X_test, y_test, lamda=11.26):
        W = self.ridge_regression(X_train, y_train, lamda)
        Ein = self.cal_error(X_train, y_train, W)
        Eout = self.cal_error(X_test, y_test, W)

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

        for i in range(len(lamda_log)):
            W = self.ridge_regression(X_Dtrain, y_Dtrain, lamda=pow(10, lamda_log[i]))  # calculate W by closed form solution
            Eval = self.cal_error(X_Dval, y_Dval, W)  # calculate Eval
            Eval_all.append(Eval)
            if Eval < Eval_best:
                Eval_best = Eval  # choose min Eval
                W_best = W
                lamda_best = pow(10, lamda_log[i])

        Eout = self.cal_error(X_test, y_test, W_best)

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
        for l in range(len(lamda_log)):
            Eval = 0.0
            for i in range(k):
                X_train = np.concatenate((X[:i*num_flod], X[(i+1)*num_flod:]), axis=0)
                y_train = np.concatenate((y[:i*num_flod], y[(i+1)*num_flod:]), axis=0)
                X_val = X[i*num_flod:(i+1)*num_flod]
                y_val = y[i*num_flod:(i+1)*num_flod]
                W = self.ridge_regression(X_train, y_train, lamda=pow(10, lamda_log[l]))
                Eval += self.cal_error(X_val, y_val, W)
            Eval /= k  # average Eval
            Eval_all.append(Eval)
            if Eval < Eval_best:
                Eval_best = Eval
                W_best = W
                lamda_best = pow(10, lamda_log[l])
                
        return lamda_best, Eval_best, Eval_all







