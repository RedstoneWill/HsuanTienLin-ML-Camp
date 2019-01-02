# First, we use an artificial data set to study PLA. 
# The data set is in https://www.csie.ntu.edu.tw/~htlin/course/ml15fall/hw1/hw1_15_train.dat 
# Each line of the data set contains one (xn ,yn ) with xn ∈ R4 . 
# The first 4 numbers of the line contains the components of x n orderly, the last number is y n . 
# Please initialize your algorithm with w = 0 and take sign(0) as −1. 
# As a friendly reminder, remember to add x0 = 1 as always!

import numpy as np

class pla(object):

    def __init__(self):
        pass
    
    # Q1. Implement a version of PLA by visiting examples in the naive cycle using the order of examples in the data set. 
    # Run the algorithm on the data set. 
    # What is the number of updates before the algorithm halts? 
    def pla_1(self, X, Y):

        # weights initialization
        W = np.zeros(X.shape[1])

        # PLA iteration
        halt = 0 # number of iteration before halt

        ### YOUR CODE HERE



        ### END YOUR CODE
        
        return halt

    # Q2. Implement a version of PLA by visiting examples in fixed, pre-determined random cycles throughout the algorithm. 
    # Run the algorithm on the data set. Please repeat your experiment for 2000 times, each with a different random seed. 
    # What is the average number of updates before the algorithm halts? 
    # Plot a histogram ( https://en.wikipedia.org/wiki/Histogram ) to show the number of updates versus frequency.
    def pla_2(self, X, Y):

        Iteration = 2000 # number of iteration
        Halts = [] # list store halt every iteration
        Accuracys = [] # list store accuracy every iteration

        ### YOUR CODE HERE



        ### END YOUR CODE

        # mean
        halt_mean = np.mean(Halts)
        accuracy_mean = np.mean(Accuracys)
    
        return halt_mean, accuracy_mean
    
    # Q3. Implement a version of PLA by visiting examples in fixed, pre-determined random cycles throughout the algorithm, while changing the update rule to be:
    # Wt+1→Wt+ηyn(t)xn(t) with  η=0.5η=0.5 . Note that your PLA in the previous problem corresponds to  η=1η=1 . 
    # Please repeat your experiment for 2000 times, each with a different random seed. What is the average number of updates before the algorithm halts? 
    # Plot a histogram to show the number of updates versus frequency. Compare your result to the previous problem and briefly discuss your findings.
    def pla_3(self, X, Y):
        
        Iteration = 2000 # number of iteration
        Halts = [] # list store halt every iteration
        Accuracys = [] # list store accuracy every iteration

        ### YOUR CODE HERE



        ### END YOUR CODE
    
        # mean
        halt_mean = np.mean(Halts)
        accuracy_mean = np.mean(Accuracys)

        return halt_mean, accuracy_mean


# Next, we play with the pocket algorithm. Modify your PLA in Problem 16 to visit examples purely randomly, 
# and then add the ‘pocket’ steps to the algorithm. We will use 
# https://www.csie.ntu.edu.tw/~htlin/course/ml15fall/hw1/hw1_18_train.dat as the training data set D, 
# and https://www.csie.ntu.edu.tw/~htlin/course/ml15fall/hw1/hw1_18_test.dat 
# as the test set for “verifying” the g returned by your algorithm (see lecture 4 about verifying). 
# The sets are of the same format as the previous one.

class pocket_pla(object):

    def __init__(slef):
        pass

    
    # Q1. Run the pocket algorithm with a total of 50 updates on D, and verify the performance of w pocket using the test set. 
    # Please repeat your experiment for 2000 times, each with a different random seed. 
    # What is the average error rate on the test set? Plot a histogram to show error rate versus frequency.

    # calculate error count
    def calError(self, X, Y, W):
        score = np.dot(X, W)
        Y_pred = np.ones_like(Y)
        Y_pred[score < 0] = -1
        err_cnt = np.sum(Y_pred != Y)
        return err_cnt

    def pocket_pla_1(self, X_train, Y_train, X_test, Y_test):
        Iteration = 2000 # number of iteration
        Update = 50
        Errors = [] # list store error rate every iteration

        ### YOUR CODE HERE



        ### END YOUR CODE
    
        # mean of errors
        error_mean = np.mean(Errors)

        return error_mean

    # Q2. Modify your algorithm to return  w50w50 (the PLA vector after 50 updates) instead of w (the pocket vector) after 50 updates. 
    # Run the modified algorithm on D, and verify the performance using the test set. 
    # Please repeat your experiment for 2000 times, each with a different random seed. 
    # What is the average error rate on the test set? Plot a histogram to show error rate versus frequency. 
    # Compare your result to the previous problem and briefly discuss your findings.
    def pocket_pla_2(self, X_train, Y_train, X_test, Y_test):
        Iteration = 2000 # number of iteration
        Update = 50
        Errors = [] # list store error rate every iteration


        ### YOUR CODE HERE



        ### END YOUR CODE
    
        # mean of error
        error_mean = np.mean(Errors)

        return error_mean

    # Q3. Modify your algorithm in Problem 1 to run for 100 updates instead of 50, and verify the performance of w pocket using the test set. 
    # Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set? Plot a 
    # histogram to show error rate versus frequency. Compare your result to Problem 18 and briefly discuss your findings.
    def pocket_pla_3(self, X_train, Y_train, X_test, Y_test):
        Iteration = 2000 # number of iteration
        Update = 100
        Errors = [] # list store error rate every iteration


        ### YOUR CODE HERE



        ### END YOUR CODE
    
        # mean of errors
        error_mean = np.mean(Errors)

        return error_mean

