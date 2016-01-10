import mnist_loader
import Network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network.Network([784, 30, 10])
X, Y = zip(*training_data)
net.gradient_descent(training_data, 10, 1, 1)
net.evaluate(training_data)
