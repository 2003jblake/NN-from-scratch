'''This module allows you to build your own custom neural networks, using a mixture of activation functions
aswell as layers and nodes'''

import numpy as np

class NeuralNetwork():
    '''This class defines all methods and function used in creating, training and predicting a neural network'''

    def relu(self, z):
        '''ReLU function'''

        return max(0, z)

    def d_relu(self, z):
        '''Derivative of ReLU function'''

        if z > 0:
            return 1

        return 0


    def leaky_relu(self, z, alpha=0.1):
        ''' Leaky ReLU function'''

        return max(alpha * z, z)

    def d_leaky_relu(self, z, alpha=0.1):
        '''Derivative of leaky ReLU function'''

        if z > 0:
            return 1

        return alpha


    def elu(self, z, alpha=1):
        '''Exponential linear unit function'''

        if z >= 0:
            return z

        return alpha * (np.exp(z) - 1)

    def d_elu(self, z, alpha=1):
        '''Derivative of ELU function'''

        if z > 0:
            return 1

        return alpha*np.exp(z)


    def sigmoid(self, z):
        '''Sigmoid function'''

        return 1/(1 + np.exp(-z))

    def d_sigmoid(self, z):
        '''Derivative of sigmoid function'''

        return self.sigmoid(z) * (1 - self.sigmoid(z))


    def tanh(self, z):
        '''tanh function'''

        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def d_tanh(self, z):
        '''derivative of tanh function'''

        return 1 - self.tanh(z)**2
