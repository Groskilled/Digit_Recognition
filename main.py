import numpy as np
import random
import Network
import scipy.io as sio

net = Network.Network([400, 30, 10])
data = sio.loadmat('ex4data1.mat')
X = np.array(data['X'])
Y = np.array(data['y'])
y_ = []
print np.array([[9 == j] for j in range(10)], dtype=int)
#for i in range(len(Y)):
#    y.append()
#net.gradient_descent(self, data, size, max_iter, lrate):
