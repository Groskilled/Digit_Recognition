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

    def cost_function(self, X, Y):
        cost = 0
        m = X.shape[0]
        for n in range(self.n_lay - 1):
            X = functions.sigmoid(np.dot(X, self.weights[n]) + self.biases[n].T)
        left = -Y * np.log(X)
        right = (1 - Y) * np.log(1 - X)
        tmp = left - right
        return np.mean(tmp)

    def back_prop(self, X, Y, lrate):
        '''
        starting with a shitty implementation working with a special network and we will improve that later
        '''
        activation = {}
        m = X.shape[0]
        for n in range(self.n_lay - 1):
            activation[n] = X
            X = functions.sigmoid(np.dot(X, self.weights[n]) + self.biases[n].T)
        delta3 = X - Y
        delta2 = np.dot(delta3,self.weights[1].T) * functions.sigmoid_grad(activation[1])
        Delta1 = np.dot(activation[0].T, delta2)
        Delta2 = np.dot(activation[1].T, delta3)
        self.weights[0] = Delta1 / m + ((lrate / m) * self.weights[0])
        self.weights[1] = Delta2 / m + ((lrate / m) * self.weights[1])

    def gradient_descent(self, data, lrate):
        '''
        data is the whole set of data
        lrate is the learning rate
        '''
        n = len(data)
    
    def __init__(self, sizes):
        self.n_lay = len(sizes)
        self.biases = self.init_biases(sizes)
        self.weights = self.init_weights(sizes)
