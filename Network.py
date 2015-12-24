import numpy as np

class Network(object):

    def init_biases(self, sizes):
        ret = {}
        for i in range(len(sizes[1:])):
            ret[i] = np.random.randn(sizes[i + 1], 1)
            print ret[i].shape
        return ret;

    def init_weights(self, sizes):
        ret = {}
        for i in range(len(sizes) - 1):
            ret[i] = np.random.randn(sizes[i], sizes[i + 1])
            print ret[i].shape
        return ret
    
    def __init__(self, sizes):
        self.n_lay = len(sizes)
        self.biases = self.init_biases(sizes)
        self.weights = self.init_weights(sizes)
