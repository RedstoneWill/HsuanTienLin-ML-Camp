import numpy as np

class LinearRegression(object):

    def __init__(self):
        pass
    
    # Q1: Carry out Linear Regression without transformation, i.e., with feature vector:(1,x1 ,x2),
    # Run the experiments for 1000 times. What is the average E in over 1000 experiments?
    
    # generate target function: f(x1, x2) = sign(x1*x1 + x2*x2 - 0.6)
    def target_function(self, X1, X2):
        y = np.ones(X1.shape[0])
        y[(X1*X1 + X2*X2 - 0.6) < 0] = -1
    
        return y
    

    # create train data with 10% flipping noise
    def generate_data_1d(self, time_seed):
        np.random.seed(time_seed)
        X = np.zeros((1000, 3))  # 1000 train data
        y = np.zeros((1000, 1))
    
        # X features
        ### YOUR CODE HERE



        ### END YOUR CODE
        # y label without noise
        y = self.target_function(X[:, 1], X[:, 2])
        # y label with  10% noise
        y_noise = y * np.where(np.random.random(1000) < 0.1, -1, 1)
    
        return X, y_noise
    

    # calculate error rate
    def cal_error(self, X, y, W):
        # calculate scores
        scores = np.dot(X, W) * y
        # calculate all errors
        error = np.sum(np.where(scores < 0, 1, 0))
        error /= X.shape[0]
    
        return error
    

    # calculate linear regression closed form solution
    def LinearR_closed_form(self, X, y):
    
        # linear regression closed form solution
        ### YOUR CODE HERE



        ### END YOUR CODE
        


    # calculate the average  Ein  by 1000 iteration
    def cal_Ein_1d(self):
        
        ### YOUR CODE HERE



        ### END YOUR CODE

        return Ein

    
    # Q2: Carry out Linear Regression without transformation, i.e., with feature vector:(1, x1, x2, x1x2, x1*x1, x2*x2),
    # Run the experiments for 1000 times. What is the average E in over 1000 experiments?

    # create train data with 10% flipping noise
    def generate_data_2d(self, time_seed):

        np.random.seed(time_seed)
        X = np.zeros((1000, 6))  # 1000 train data
        y = np.zeros((1000, 1))
    
        # X features
        ### YOUR CODE HERE



        ### END YOUR CODE
        # y label without noise
        y = self.target_function(X[:, 1], X[:, 2])
        # y label with  10% noise
        y_noise = y * np.where(np.random.random(1000) < 0.1, -1, 1)
    
        return X, y_noise

    
    # calculate the average  Ein  by 1000 iteration
    def cal_Ein_2d(self):
        
        ### YOUR CODE HERE



        ### END YOUR CODE

        return Ein
    

class LogisticRegression(object):

    def __init__(self):
        pass

    # Implement the fixed learning rate gradient descent algorithm below for logistic regression, 
    # initialized with 0. Run the algorithm with η = 0.001 and T = 2000. What is the Eout from your algorithm, 
    # evaluated using the 0/1 error on the test set?
    
    # define load data function
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

    
    # define sigmoid function
    def sigmoid(self, x):
        
        ### YOUR CODE HERE



        ### END YOUR CODE
    

    # Q1: Gradient Descent
    # define gradient descent function
    def gradient_descent(self, X, y):
        y = y.reshape(-1, 1)  # reshape (1000,) to (1000,1)
        m = X.shape[0]  # number of samples
        n = X.shape[1]  # number of features
        T = 2000  # number of iteration
        learning_rate = 0.001  # learning rate
        W = np.zeros((n, 1))  # initialize weights
    
        for i in range(T):
            ### YOUR CODE HERE



            ### END YOUR CODE
            

        return W
    

    # define predict error function by W
    def predict(self, X, y, W):
        y_hat = self.sigmoid(X.dot(W))  # predict probability [0,1]
        y_hat[y_hat >= 0.5] = 1  # positive
        y_hat[y_hat < 0.5] = -1  # negative
    
        y = y.reshape(-1,1)  # reshape 2D
        Ein_mean = np.mean(y_hat != y)

        return Ein_mean
    

    # Logistic regression by GD
    def lr_gd(self, X_train, y_train, X_test, y_test):
        # Gradient descent
        W = self.gradient_descent(X_train, y_train)  # calculate weights

        # calculate test data error
        Ein_mean = self.predict(X_test, y_test, W)

        return Ein_mean


    # Q2: Stochastic Gradient Descent
    # define stochastic gradient descent function
    def stochastic_gradient_descent(self, X, y):
        m = X.shape[0]  # number of samples
        n = X.shape[1]  # number of features
        T = 2000  # number of iteration
        learning_rate = 0.001  # learning rate
        W = np.zeros((n, 1))  # initialize weights
    
        for t in range(T):
            ### YOUR CODE HERE



            ### END YOUR CODE

        return W

    # Logistic regression by SGD
    def lr_sgd(self, X_train, y_train, X_test, y_test):
        # Gradient descent
        W = self.stochastic_gradient_descent(X_train, y_train)  # calculate weights

        # calculate test data error
        Ein_mean = self.predict(X_test, y_test, W)

        return Ein_mean















