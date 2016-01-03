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

    def feedforward(self, a):
        for n in range(self.n_lay - 1):
            a = functions.sigmoid(np.dot(self.weights[n].T, a) + self.biases[n])
        return a

    def cost_function(self, X):
        cost = 0
        for ex in X:
            est = self.feedforward(np.reshape(ex[0:400], (400, 1)))
            left = np.reshape(ex[400:410], (10, 1)) - np.log(est)
            right = (1 - np.reshape(ex[400:410], (10, 1))) * np.log(1 - est)
            cost = cost + np.sum(left - right) / 100
        cost = cost / X.shape[0]
        return cost

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
