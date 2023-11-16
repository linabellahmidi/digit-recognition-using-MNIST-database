import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("C:\\Users\\lina\\OneDrive\\Desktop\\mnist")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *


# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
#Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])




def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error



print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=0.01))
