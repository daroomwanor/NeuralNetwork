import os
import numpy as np


class NeuralNetwork:
    
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

        print(self.weights1)

    def sigmoid(self, x):
        return 1 /(1+np.exp(-x))


    def sigmoid_derivative(self, x):
        return 1 / (1 + np.exp(-x)) * (1 - (1 / (1 + np.exp(-x))))


if __name__ == "__main__":
    x = np.zeros((4,4))
    y = np.zeros((4,1))
    node = NeuralNetwork(x, y)
    node.feedforward()
    node.backprop()