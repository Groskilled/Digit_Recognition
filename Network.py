from __future__ import division
import functions
import random
import numpy as np

class Network(object):

    def feed_forward(self, X):
        for b, w in zip(self.biases, self.weights):
            X = np.dot(w.T, X) + b
        return X

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        a = x
        acts = [x]
        zs = [] 
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w.T, a) + b
            zs.append(z)
            a = functions.sigmoid(z)
            acts.append(a)
        delta = (a - y) * functions.sigmoid_grad(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, acts[-2].T).T
        for i in xrange(2, self.n_lay):
            z = zs[-i]
            delta = np.dot(self.weights[-i + 1], delta) * functions.sigmoid_grad(z)
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, acts[-i - 1].T).T
        return nabla_b, nabla_w

    def update(self, batch, lrate):
        m = len(batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            dnabla_b, dnabla_w = self.backprop(x,y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, dnabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, dnabla_w)]
        self.weights = [w - (lrate/m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (lrate/m) * nb for b, nb in zip(self.biases, nabla_b)]

    def gradient_descent(self, data, size, cicles, lrate, test_data=None):
        m = len(data)
        if test_data: n = len(test_data)
        for i in xrange(cicles):
            random.shuffle(data)
            batches = [data[k: k+size] for k in xrange(0,m,size)]
            for batch in batches: self.update(batch, lrate)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n)

    def evaluate(self, X):
        total = 0
        for x, y in X:
            if y == np.argmax(self.feed_forward(x)):
                total = total + 1
        return total

    def __init__(self, sizes):
        self.n_lay = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
