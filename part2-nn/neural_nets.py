import numpy as np
import math


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    # TODO
    if (x>0):
        return x
    else:
        return 0

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    # TODO
    if (x>0):
        return 1
    else:
        return 0

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS (Initialized to floats instead of ints)
        self.input_to_hidden_weights = np.matrix('1. 1.; 1. 1.; 1. 1.')
        self.hidden_to_output_weights = np.matrix('1. 1. 1.')
        self.biases = np.matrix('0.; 0.; 0.')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]

    def train(self, x1, x2, y):

        ### Forward propagation ###
        input_values = np.array([[x1],[x2]]) # 2 by 1
        print("input values : ", input_values)

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = self.input_to_hidden_weights
        vectorized = np.vectorize(rectified_linear_unit)
        Z = np.matmul(hidden_layer_weighted_input,input_values)+self.biases
        print("Z : ", Z)
        hidden_layer_activation = vectorized(Z)
        print("reLU de Z : ", hidden_layer_activation)

        output =  np.sum(np.multiply(hidden_layer_activation,np.transpose(self.hidden_to_output_weights)))
        print ("output : ", output)
        activated_output = output_layer_activation(output)
        print ("activated_output : ", activated_output)

        ### Backpropagation ###

        # Compute gradients
        output_layer_error = (activated_output-y)*output_layer_activation_derivative(output)
        print ("output_layer_error : ", output_layer_error)
        vectorized_rectified_linear_unit_derivative = np.vectorize(rectified_linear_unit_derivative)
        hidden_layer_error = output_layer_error*(np.multiply(vectorized_rectified_linear_unit_derivative(Z),np.transpose(self.hidden_to_output_weights))) #TODO (3 by 1 matrix)
        print ("hidden_layer_error : ", hidden_layer_error)

        bias_gradients = hidden_layer_error #TODO
        hidden_to_output_weight_gradients = np.transpose(output_layer_error*hidden_layer_activation)
        print ("hidden_to_output_weight_gradients : ", hidden_to_output_weight_gradients)
        input_to_hidden_weight_gradients = np.matmul(hidden_layer_error,np.transpose(input_values))
        print ("input_to_hidden_weight_gradients : ", input_to_hidden_weight_gradients)

        # Use gradients to adjust weights and biases using gradient descent
        self.biases -= self.learning_rate*bias_gradients
        print("new bias : ", self.biases)
        self.input_to_hidden_weights -= self.learning_rate*input_to_hidden_weight_gradients 
        print("new input_to_hidden_weights : ", self.input_to_hidden_weights)
        self.hidden_to_output_weights -= self.learning_rate*hidden_to_output_weight_gradients
        print("new hidden_to_output_weights : ", self.hidden_to_output_weights) 

    def predict(self, x1, x2):

        input_values = np.matrix([[x1],[x2]])

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = np.matmul(self.input_to_hidden_weights,input_values)
        vectorized_rectified_linear_unit = np.vectorize(rectified_linear_unit)
        hidden_layer_activation = vectorized_rectified_linear_unit(hidden_layer_weighted_input)
        output = np.dot(np.transpose(hidden_layer_activation),self.hidden_to_output_weights.A1)
        activated_output = output_layer_activation(output)

        return activated_output.item()


    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return

x = NeuralNetwork()

#x.train_neural_network()
x.test_neural_network()
