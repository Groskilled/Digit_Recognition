import numpy as np
import random
import time
import functions

class Network(object):

    def init_biases(self, sizes):
        ret = {}
        for i in range(len(sizes[1:])):
            ret[i] = np.random.randn(sizes[i + 1], 1)
        return ret;

    def init_weights(self, sizes):
        ret = {}
        for i in range(len(sizes) - 1):
            ret[i] = np.random.randn(sizes[i], sizes[i + 1])
        return ret

    def cost_function(self, X, Y):
        cost = 0
        m = X.shape[0]
        for n in range(self.n_lay - 1):
            X = functions.sigmoid(np.dot(X, self.weights[n]) + self.biases[n].T)
        left = -Y * np.log(X)
        right = (1 - Y) * np.log(1 - X)
        tmp = left - right
        return np.mean(tmp)

    def gradient_descent(self, data, size, max_iter, lrate):
        '''
        data is the whole set of data
        size is the desired size for the training set
        max_iter is the number of time we want to train
        lrate is the learning rate
        '''
        n = len(data)
        for i in range(max_iter):
            random.shuffle(data)
            mini_batch = np.array([data[k] for k in range(size)])
            self.cost_function(mini_batch)
            print "Cycle number {0} complete.".format(i)
    
    def __init__(self, sizes):
        self.n_lay = len(sizes)
        self.biases = self.init_biases(sizes)
        self.weights = self.init_weights(sizes)
