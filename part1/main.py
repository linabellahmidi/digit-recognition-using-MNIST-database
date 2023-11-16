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
# Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])

#######################################################################
#Linear Regression with Closed Form Solution
#######################################################################



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



print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=1))


#######################################################################
#Support Vector Machine
#######################################################################


def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())


def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
#Multinomial (Softmax) Regression and Gradient Descent
#######################################################################


def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    #plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")

    train_y_mod3,test_y_mod3 = update_y(train_y, test_y)
    test_error_mod3=compute_test_error_mod3(test_x,test_y_mod3,theta,temp_parameter=1)
    print(test_error_mod3)
    return test_error


print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=0.5))
print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))
print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=2))


#######################################################################
#Changing Labels
#######################################################################



def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y_mod3,test_y_mod3 = update_y(train_y, test_y)
    theta, cost_function_history = softmax_regression(train_x, train_y_mod3, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    #plot_cost_function_over_time(cost_function_history)
    test_error_mod3 = compute_test_error(test_x, test_y_mod3, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")
    return test_error_mod3


print('softmax test_error=', run_softmax_on_MNIST_mod3(temp_parameter=1))

#######################################################################
#Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##

n_components = 18


train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)
train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

# train_pca (and test_pca) is a representation of our training (and test) data
# after projecting each example onto the first 18 principal components.


# Training softmax regression model using (train_pca, train_y)
#       and evaluating its accuracy on (test_pca, test_y).
train_x, train_y, test_x, test_y = get_MNIST_data()
theta, cost_function_history = softmax_regression(train_pca, train_y, temp_parameter=1, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
#plot_cost_function_over_time(cost_function_history)
test_error_pca = compute_test_error(test_pca, test_y, theta, temp_parameter=1)
# Save the model parameters theta obtained from calling softmax_regression to disk.
write_pickle_data(theta, "./theta.pkl.gz")
print("Test error : ", test_error_pca)

#       produce scatterplot of the first 100 MNIST images, as represented in the space spanned by the
#       first 2 principal components found above.
plot_PC(train_x[range(000, 100), ], pcs, train_y[range(000, 100)], feature_means)#feature_means added since release


firstimage_reconstructed = reconstruct_PC(train_pca[0, ], pcs, n_components, train_x, feature_means)#feature_means added since release
plot_images(firstimage_reconstructed)
plot_images(train_x[0, ])

secondimage_reconstructed = reconstruct_PC(train_pca[1, ], pcs, n_components, train_x, feature_means)#feature_means added since release
plot_images(secondimage_reconstructed)
plot_images(train_x[1, ])