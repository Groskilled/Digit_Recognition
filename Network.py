import numpy as np
import random
import time
import functions

class Network(object):

    def init_biases(self, sizes):
        ret = {}
        for i in range(len(sizes[1:])):
            #ret[i] = np.random.randn(sizes[i + 1], 1)
            #for now we are using ones as biases
            ret[i] = np.ones((sizes[i+1], 1))
        return ret;

    def init_weights(self, sizes):
        ret = {}
        for i in range(len(sizes) - 1):
            ret[i] = np.random.randn(sizes[i], sizes[i + 1])
        return ret

    def set_weights(self, T1, T2):
        self.weights[0] = T1
        self.weights[1] = T2

    def cost_function(self, X, Y):
        for n in range(self.n_lay - 1):
            X = np.c_[np.ones((X.shape[0], 1)), X]
            X = functions.sigmoid(np.dot(X, self.weights[n].T))
        left = -Y * np.log(X)
        right = (1 - Y) * np.log(1 - X)
        tmp = np.sum(left - right, axis=1)
        return np.mean(tmp)

    def back_prop(self, X, Y, lrate):
        '''
        starting with a shitty implementation working with a special network and we will improve that later
        '''
        activation = {}
        m = X.shape[0]
        for n in range(self.n_lay - 1):
            activation[n] = X
            X = functions.sigmoid(np.dot(X, self.weights[n]))
        delta3 = X - Y
        delta2 = np.dot(delta3,self.weights[1].T) * functions.sigmoid_grad(activation[1])
        Delta1 = np.dot(activation[0].T, delta2)
        Delta2 = np.dot(activation[1].T, delta3)
        self.weights[0] = (Delta1 / m) - ((lrate / m) * self.weights[0])
        self.weights[1] = (Delta2 / m) - ((lrate / m) * self.weights[1])

    def evaluate(self, X, Y):
        m = X.shape[0]
        right = 0
        for n in range(self.n_lay - 1):
            X = functions.sigmoid(np.dot(X, self.weights[n]))
        print X[0]
        #for i in range(m - 1):
        #    np.array_equal(X[i], Y[i])

    def gradient_descent(self, data, cicles, lrate):
        '''
        data is the whole set of data
        lrate is the learning rate
        '''
        n = len(data)
        size = int(n * 0.6)
        print self.cost_function(data[0:n, 0:400], data[0:n, 400:410])
        for i in range(cicles):
            random.shuffle(data)
            X = data[0:size,0:400]
            Y = data[0:size,400:410]
            self.back_prop(X, Y, lrate)
        self.evaluate(data[size:n,0:400], data[size:n, 400:410])
        print self.cost_function(X, Y)
    
    def __init__(self, sizes):
        self.n_lay = len(sizes)
        self.biases = self.init_biases(sizes)
        self.weights = self.init_weights(sizes)
