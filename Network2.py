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

    def back_prop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        a = np.reshape(x,(x.shape[0],1))
        y = np.reshape(y,(y.shape[0],1))
        zs = []
        acts = [a]
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w.T, a) + b
            zs.append(z)
            a = functions.sigmoid(z)
            acts.append(a)
        delta = (a-y) * functions.sigmoid_grad(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(acts[-2], delta.T)
        for i in xrange(2, self.n_lay):
            delta = np.dot(self.weights[-i+1], delta) * functions.sigmoid_grad(zs[-i])
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(acts[-i-1], delta.T)
        return (nabla_b, nabla_w)

    def update(self, batch, lrate):
        x, y = zip(*batch)
        m = len(x)
        x = np.array([i.T[0] for i in x])
        y = np.array([i.T[0] for i in y])
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for a, b in zip(x,y):
            dnabla_b, dnabla_w = self.back_prop(a, b)
            nabla_b = [nb + dnb for nb,dnb in zip(nabla_b, dnabla_b)]
            nabla_w = [nw + dnw for nw,dnw in zip(nabla_w, dnabla_w)]
        self.biases = [b - (lrate / m) * nb for b,nb in zip(self.biases, nabla_b)]
        self.weights = [w - (lrate / m) * nw for w,nw in zip(self.weights, nabla_w)]

    def gradient_descent(self, data, cicles, size, lrate, test_set=None):
        n = len(data)
        if test_set: n_test = len(test_set)
        for i in xrange(cicles):
            random.shuffle(data)
            batches = [data[k:k+size] for k in xrange(0, n, size)]
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
