'''This module allows you to build your own custom neural networks, using a mixture of activation functions
aswell as layers and nodes'''

import logging
import numpy as np
import preprocessor as p


def log_loss_function(y_pred, y_targ):
    '''This calculates loss using the logarithmic loss function
    aka the cross-entropy loss function'''

    y_pred = min(y_pred, 0.99999999)
    y_pred = max(y_pred, 0.00000001)

    loss = -(y_targ * np.log(y_pred) + (1 - y_targ) * np.log(1 - y_pred))

    return loss

def d_log_loss_function(y_pred, y_targ):
    '''The derivative of the logarithmic loss function'''

    d_loss = y_targ 


def squared_loss(y_pred, y_targ):
    '''squared loss function'''

    d_loss = (y_targ - y_pred)**2

    return d_loss

def d_squared_loss(y_pred, y_targ):
    '''The derivative of the squared loss function'''

    loss = 2*(y_pred - y_targ)

    return loss

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


def get_d_act_func(a_f):
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

    logging.error('could not return corresponding derivative of activation function')
    return None

def get_d_loss_func(l_f):
    '''gets the corresponding derivative of the loss function passed in'''

    if l_f is squared_loss:
        return d_squared_loss
    if l_f is log_loss_function:
        return d_log_loss_function

    logging.error('could not return corresponding derivative of loss function')
    return None



class Layer():
    '''repersents a layer in the neural network, where you can pass in the activation function used
    the input and output size'''

    def __init__(self, input_size, num_neurons, act_func=relu):
        self.act_func = np.vectorize(act_func)
        self.d_act_func = np.vectorize(get_d_act_func(act_func))
        self.input_size = input_size
        self.num_neurons = num_neurons


        self.bias = np.asmatrix(np.zeros(self.num_neurons))
        self.weights = np.random.rand(self.input_size, self.num_neurons) - 0.5


    def forw_pass(self, x):
        '''This peforms a forward pass through the function given the input data  x
        returns the activation of each neuron in the layer'''

        self.input = x
        x = np.dot(x, self.weights) + self.bias

        self.output = x

        return self.act_func(x)
 

    def back_pass(self, prev_delta, lr=0.1):
        '''This peforms  single back pass of a layer, takes in the delta of  the previous
        layer and returns the delta, updates weights and biases of the current layer'''


        error = prev_delta * (self.d_act_func(self.output))

        weights_grad = (self.input.T).dot(error)

        self.weights -= (lr * weights_grad)
        self.bias -= lr * error

        delta = ((self.input.T).dot(error)).sum()

        return delta



class NeuralNetwork():
    '''This class defines all methods and function used in creating, training and predicting a neural network'''

    def __init__(self, loss_func=squared_loss):
        self.layers = []
        self.loss_func = loss_func
        self.d_loss_func = get_d_loss_func(loss_func)


    def add_layer(self, layer):
        '''Adds a layer to the nerual network, adds it after all the previously added
        layers'''
        self.layers.append(layer)
    

    def forw_prop(self, data_instance):
        '''peforms forward propergation of the whole netowrk for a single training data instance'''
        features = data_instance

        for j in range(len(self.layers)):
            features = self.layers[j].forw_pass(features)

        return features


    def back_prop(self, delta):
        '''peforms back propergation of the whole network for a single training data instance'''
        
        for j in range(len(self.layers) - 1):
            delta = self.layers[len(self.layers)-1 - j].back_pass(delta)


    def train(self ,x_train, y_train):
        '''Takes in training features, and training target, peforms forward and back propergation
        to train the neural network'''

        num_rows, _ = x_train.shape

        for x in range(10):
            for i in range(num_rows):

                pass_output = self.forw_prop(x_train[i])
                delta = self.d_loss_func(pass_output, y_train[i])    
                self.back_prop(delta)


    def predict(self ,x_test, y_test):

        x = 0

        num_rows, _ = x_test.shape
        for i in range(num_rows):

            features = x_test[i]

            for j in range(len(self.layers)):
                features = self.layers[j].forw_pass(features)
            
            if round(features[0,0]) == y_test[i]:
                x += 1

        return(x)
