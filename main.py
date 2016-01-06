import numpy as np
import random
import Network
import scipy.io as sio

net = Network.Network([400, 25, 10])
data = sio.loadmat('ex4data1.mat')
theta = sio.loadmat('ex4weights.mat')
T1 = np.array(theta['Theta1'])
T2 = np.array(theta['Theta2'])
net.set_weights(T1, T2)
X = np.array(data['X'])
Y = np.array(data['y'])
y_ = []
for i in range(len(Y)):
    y_.append(np.array([[Y[i] == j] for j in range(1, 11)], dtype=int).T[0][0])
y_ = np.array(y_)
X_ = np.c_[X, y_]
net.evaluate(X,y_)
net.gradient_descent(X_, 1, 0.01)
net.gradient_descent(X_, 10, 0.01)
net.gradient_descent(X_, 100, 0.01)
net.evaluate(X,y_)
net.gradient_descent(X_, 1000, 0.01)
net.evaluate(X,y_)
