import numpy as np

a=np.sum(np.multiply(np.array([[3.],[3.],[3.]]),np.transpose(np.matrix('1. 1. 1.'))))
a = np.array([[3.], [3.], [3.]])  # Create a 3x1 array
b = np.array([1., 1., 1.])  # Create a 1D array

result = np.dot(a.T, b)  # Transpose a and take the dot product
print("Dot product:", result)