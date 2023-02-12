'''This module allows you to build your own custom neural networks, using a mixture of activation functions
aswell as layers and nodes'''

import logging
import numpy as np


class NeuralNetwork():
    '''This class defines all methods and function used in creating, training and predicting a neural network'''

    def log_loss_function(self, y_pred, y_targ):
        '''This calculates loss using the logarithmic loss function
        aka the cross-entropy loss function'''
        
        if y_pred > 0.99999999:
            y_pred = 0.99999999
        if y_pred < 0.00000001:
            y_pred 
    

def relu(z):
    '''ReLU function'''

    return max(0, z)

def d_relu(z):
    '''Derivative of ReLU function'''

    if z > 0:
         return 1

    return 0


def leaky_relu(z, alpha=0.1):
    ''' Leaky ReLU function'''

    return max(alpha * z, z)

def d_leaky_relu(z, alpha=0.1):
    '''Derivative of leaky ReLU function'''

    if z > 0:
        return 1

    return alpha


def elu(z, alpha=1):
    '''Exponential linear unit function'''

    if z >= 0:
        return z

    return alpha * (np.exp(z) - 1)

def d_elu(z, alpha=1):
    '''Derivative of ELU function'''

    if z > 0:
        return 1

    return alpha*np.exp(z)


def sigmoid(z):
    '''Sigmoid function'''

    return 1/(1 + np.exp(-z))

def d_sigmoid(z):
    '''Derivative of sigmoid function'''

    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    '''tanh function'''

    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def d_tanh(z):
    '''derivative of tanh function'''

    return 1 - tanh(z)**2

def get_d_func(a_f):
    '''Gets the corresponding derivative of the activation function passed in'''

    if a_f is relu:
        return d_relu
    if a_f is leaky_relu:
        return d_leaky_relu
    if a_f is elu:
        return d_elu
    if a_f is sigmoid:
        return d_sigmoid
    if a_f is tanh:
        return d_tanh

    logging.error('could not return corresponding derivative function')
    return None

class Layer():
    '''repersents a layer in the neural network, where you can pass in the activation function used
    the input and output size'''

    def __init__(self, input_size, num_neurons, act_func=relu):
        self.act_func = act_func
        self.d_act_func = get_d_func(self.act_func)
        self.input_size = input_size
        self.num_neurons = num_neurons

        self.bias = np.zeros(self.num_neurons)
        self.weights = np.random.rand(self.input_size, self.num_neurons) - 0.5


    def forw_pass(self, x):
        '''This peforms a pass through the function given the input data  x'''

        x = x.dot(self.weights) + self.bias
        

    def back_prop():
        #todo
        return None

l = Layer(13, 8)
print(l.weights)