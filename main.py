import mnist_loader
import Network2
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
'''
net = Network.Network([784, 30, 10])
print "starting with {0} / 10000".format(net.evaluate(test_data))
net.gradient_descent(training_data, 50, 20, 3, test_data)
'''
net = Network2.Network2([784, 30, 10])
X, Y = zip(*training_data)
X_ = []
for i in xrange(len(X)):
    X_.append(X[i].T[0])
tmpx, tmpy = [], []
k, _ = zip(*test_data)
#mdr =  [i.T[0] for i in k]
#print np.array(mdr).shape
for j in xrange(len(test_data)):
    tmpx.append(k[j].T[0])
    tmpy.append(_[j])
tmpx = np.array(tmpx)
X = np.array(X_)
#print net.evaluate(tmpx, tmpy)
net.gradient_descent(training_data, 5, 10, 1, test_data)
