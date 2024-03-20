import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad
from torch.distributions import Normal
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Define your arbitrary function
def my_function_deac(x):
    return x**2


def my_function(x):
    sigma = 1
    mu = 0
    pdf = (1/sigma * np.sqrt(2 * np.pi)) * np.e**(-0.5 * ((x - mu)/sigma)**2)
    return pdf


# Function for adaptive integration and supporting points determination
def adaptive_integration(func, a, b, num_support_points):
    # Perform adaptive quadrature
    result, error = quad(func, a, b, epsabs=1.49e-12, epsrel=1.49e-12)

    # Determine supporting points
    supporting_points_x = np.linspace(a, b, num_support_points)
    supporting_points_y = func(supporting_points_x)

    return result, supporting_points_x, supporting_points_y

# Integration interval
a = -2
b = 2

# Number of supporting points
num_support_points = 20

# Perform adaptive integration and get supporting points
integral, supporting_points_x, supporting_points_y = adaptive_integration(my_function, a, b, num_support_points)

# Plot the original function and supporting points
x_values = np.linspace(a, b, 1000)
y_values = my_function(x_values)

plt.plot(x_values, y_values, label='Original Function')
plt.scatter(supporting_points_x, supporting_points_y, color='red', label='Supporting Points')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Function and Supporting Points')
plt.show()

print("Numerical integral:", integral)