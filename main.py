import numpy as np
import random
import Network
import scipy.io as sio

net = Network.Network([400, 30, 10])
data = sio.loadmat('ex4data1.mat')
X = np.array(data['X'])
Y = np.array(data['y'])
y_ = []
#print np.array([[1 == j] for j in range(10)], dtype=int).T[0]
for i in range(len(Y)):
    y_.append(np.array([[Y[i] % 10 == j] for j in range(10)], dtype=int).T[0][0])
y_ = np.array(y_)
#net.gradient_descent(self, data, size, max_iter, lrate):
