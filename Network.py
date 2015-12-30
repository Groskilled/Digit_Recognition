import numpy as np

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
        for n in range(self.n_lay):
            a = np.dot(self.weights[n], a)

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
            mini_set = np.array(training_data[k:k+size] for k in xrange(0, n, size))
            print mini_set.shape
            print "Cycle number {0} complete.".format(i)
    
    def __init__(self, sizes):
        self.n_lay = len(sizes)
        self.biases = self.init_biases(sizes)
        self.weights = self.init_weights(sizes)
