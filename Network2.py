from __future__ import division
import functions
import random
import numpy as np

class Network2(object):

    def feed_forward(self, X):
        for w,b in zip(self.weights, self.biases):
            X = np.dot(X, w) + b.T
        return X

    def evaluate(self, X, Y):
        total = 0
        for x, y in zip(X,Y):
            if y == np.argmax(self.feed_forward(x)):
                total = total + 1
        return total

    def update(self, batch, lrate):
        x, y = zip(*batch)
        x = np.array([i.T[0] for i in x])
        y = np.array([i.T[0] for i in y])
        dnabla_b = [np.zeros(b.shape) for b in self.biases]
        dnabla_w = [np.zeros(w.shape) for w in self.weights]
        a = x
        zs = []
        acts = []
        for w,b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b.T
            zs.append(z)
            a = functions.sigmoid(z)
            acts.append(a)
        delta = (a - y) * functions.sigmoid_grad(z)
        dnabla_b[-1] = delta
        dnabla_w[-1] = np.dot(delta.T, acts[-2]).T
        for i in xrange(2, self.n_lay):
            z = zs[-i]
            print "self.weights[-i+1].shape: {0}".format(self.weights[-i + 1].shape)
            print "delta.shape: {0}".format(delta.shape)
            print "self.biases[-i].shape {0}".format(self.biases[-i].shape)
            delta = np.dot(self.weights[-i + 1], delta.T)
            print "delta.shape after dot: {0}".format(delta.shape)
        return 1

    def gradient_descent(self, data, cicles, size, lrate, test_set=None):
        n = len(data)
        if test_set: n_test = len(test_set)
        for i in xrange(cicles):
            random.shuffle(data)
            batches = [data[0:size]]
            for batch in batches:
                self.update(batch, lrate)
            if test_set:
                x, y = zip(*test_set)
                print "Epoch {0} is over: {1} / {2}".format(i, self.evaluate([i.T[0] for i in x], y), n_test)
            else:
                print "Epoch {0} is over.".format(i)

    def __init__(self, sizes):
        self.n_lay = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
