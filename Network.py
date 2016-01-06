from __future__ import division
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
        #for i in range(len(sizes) - 1):
        #    ret[i] = np.random.randn(sizes[i]+i, sizes[i + 1]+1).T
        ret[0] = np.random.randn(25,401)
        ret[1] = np.random.randn(10,26)
        return ret

    def set_weights(self, T1, T2):
        self.weights[0] = T1
        self.weights[1] = T2

    def cost_function(self, X, Y, lrate):
        m = X.shape[0]
        for n in range(self.n_lay - 1):
            X = np.c_[np.ones((X.shape[0], 1)), X]
            X = functions.sigmoid(np.dot(X, self.weights[n].T))
        left = -Y * np.log(X)
        right = (1 - Y) * np.log(1 - X)
        tmp = np.sum(left - right, axis=1)
        left_reg = np.sum(np.power(self.weights[0][:, 2:self.weights[0].shape[1]], 2))
        right_reg = np.sum(np.power(self.weights[1][:, 2:self.weights[1].shape[1]], 2))
        return (np.mean(tmp) + (lrate / (2 * m)) * (left_reg + right_reg))

    def back_prop(self, X, Y, lrate):
        '''
        starting with a shitty implementation working with a special network and we will improve that later
        '''
        m = X.shape[0]
        '''
        activation = {}
        for n in range(self.n_lay - 1):
            X = np.c_[np.ones((X.shape[0], 1)), X]
            activation[n] = X
            X = functions.sigmoid(np.dot(X, self.weights[n].T))
        print activation[0].shape
        print activation[1].shape
        '''
        a1 = np.c_[np.ones((X.shape[0], 1)), X]
        z2 = np.dot(a1, self.weights[0].T)
        a2 = functions.sigmoid(z2)
        a2 = np.c_[np.ones((a2.shape[0], 1)),a2]
        z3 = np.dot(a2, self.weights[1].T)
        a3 = functions.sigmoid(z3)
        delta3 = a3 - Y
        delta2 = np.multiply(np.dot(delta3,self.weights[1][:,1:self.weights[1].shape[1]]), functions.sigmoid_grad(z2))
        Delta1 = np.dot(delta2.T, a1)
        Delta2 = np.dot(delta3.T, a2)
        self.weights[0][:,0] = 0
        self.weights[1][:,0] = 0
        self.weights[0] = self.weights[0] - (lrate/m) * Delta1
        self.weights[1] = self.weights[1] - (lrate/m) * Delta2
        #self.weights[0] = (Delta1 / m) + ((lrate / m) * self.weights[0])
        #self.weights[1] = (Delta2 / m) + ((lrate / m) * self.weights[1])

    def evaluate(self, X, Y):
        m = X.shape[0]
        right = 0
        for n in range(self.n_lay - 1):
            X = np.c_[np.ones((X.shape[0], 1)), X]
            X = functions.sigmoid(np.dot(X, self.weights[n].T))
        for i in range(m - 1):
            if Y[i][np.argmax(X[i]) - 1] == 1:
                right = right + 1
        print (right/m * 100)

    def gradient_descent(self, data, cicles, lrate):
        '''
        data is the whole set of data
        lrate is the learning rate
        '''
        n = len(data)
        size = n * 0.6
        for i in range(cicles):
            random.shuffle(data)
            X = data[0:size,0:400]
            Y = data[0:size,400:410]
            self.back_prop(X, Y, lrate)
    
    def __init__(self, sizes):
        self.n_lay = len(sizes)
        self.biases = self.init_biases(sizes)
        self.weights = self.init_weights(sizes)
