import numpy as np
from preprocess import *

class Decision(object):

    def __init__(self):
        pass
    
    # First,Generate a data set of size 20 by the procedure above and run the one-dimensional decisionstump algorithm on the data set. 
    # Record Ein. Repeat the experiment (including data generation, running the decision stump algorithm, and computing Ein) 5,000 times. 
    # What is the average Ein ?
    def calculate_Ein(self, X, Y):
        # calculate median of interval & negative infinite & positive infinite
        thetas = np.array([float("-inf")] + [(X[i] + X[i + 1]) / 2 for i in range(0, X.shape[0] - 1)] + [float("inf")])
        Ein = X.shape[0]
        sign = 1
        target_theta = 0.0
        # positive and negative rays
        for theta in thetas:
            y_positive = np.where(X > theta, 1, -1)
            y_negative = np.where(X < theta, 1, -1)
            error_positive = sum(y_positive != Y)
            error_negative = sum(y_negative != Y)
            if error_positive > error_negative:
                if Ein > error_negative:
                    Ein = error_negative
                    sign = -1
                    target_theta = theta
            else:
                if Ein > error_positive:
                    Ein = error_positive
                    sign = 1
                    target_theta = theta
        # two corner cases
        if target_theta == float("inf"):
            target_theta = 1.0
        if target_theta == float("-inf"):
            target_theta = -1.0
        Ein = Ein / X.shape[0] # mean of Ein
        return Ein, target_theta, sign
    
    # Repeat the experiment (including data generation, running the decision stump algorithm, and computing Ein) 5,000 times. 
    # What is the average Ein
    def decision_ray(self):
        T = 5000 # iteration
        Ein_all = [] # list for all Ein

        for i in range(T):
            X, Y = generate_input_data(i)
            Ein, theta, sign = self.calculate_Ein(X, Y)
            Ein_all.append(Ein)
            #print(('Iter = %d\t Ein = %f') % (i+1, Ein))

        # mean of Ein
        Ein_mean = np.mean(Ein_all)
        return Ein_mean
    
    # Run the algorithm on the Dtrain . What is the Ein of the optimal decision stump? 
    def decision_dtrain(self, path):
        X, Y = read_input_data(path)
        # record optimal descision stump parameters
        Ein = 1.0
        theta = 0
        sign = 1
        index = 0
        # multi decision stump optimal process
        for i in range(0, X.shape[1]):
            input_x = X[:, i]
            input_data = np.transpose(np.array([input_x, Y]))
            input_data = input_data[np.argsort(input_data[:, 0])]
            curr_Ein, curr_theta, curr_sign = self.calculate_Ein(input_data[:, 0], input_data[:, 1])
            if Ein > curr_Ein:
                Ein = curr_Ein
                theta = curr_theta
                sign = curr_sign
                index = i
        return Ein, theta, sign, index
    
    # Use the returned decision stump to predict the label of each example within the Dtest . Report an estimate Etest.
    def decision_dtest(self, path, theta, sign, index):
        # test process
        test_x, test_y = read_input_data(path)
        test_x = test_x[:, index]
        predict_y = np.array([])
        if sign == 1:
            predict_y = np.where(test_x > theta, 1.0, -1.0)
        else:
            predict_y = np.where(test_x < theta, 1.0, -1.0)
        Etest = np.mean(predict_y != test_y)
        return Etest