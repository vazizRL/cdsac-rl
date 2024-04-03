import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from scipy.integrate import quad
from torch.distributions import Normal
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Define your arbitrary function
def my_function_(x):
    return x**2


def gauss_pdf(x):
    sigma = 2
    mu = 0
    pdf = (1/sigma * np.sqrt(2 * np.pi)) * np.e**(-0.5 * ((x - mu)/sigma)**2)
    return pdf


# Function for adaptive integration and supporting points determination
def sci_integration(func, a, b, num_support_points):
    # Perform adaptive quadrature
    result, error = quad(func, a, b, epsabs=1.49e-12, epsrel=1.49e-12)

    # Determine supporting points
    supporting_points_x = np.linspace(a, b, num_support_points)
    supporting_points_y = func(supporting_points_x)

    return result, supporting_points_x, supporting_points_y


def py_integration(func, int_l, int_u, n_points):
    x = torch.linspace(start=int_l, end=int_u, steps=n_points)
    y = func(x)
    cdf = torch.trapz(y, x) + 1e-55

    return cdf


if __name__ == '__main__':
    plot = False
    # Integration interval
    # a = -3.1
    # b = 3.1
    a = -10
    b = 10

    # Number of supporting points
    num_support_points = 30

    # Perform adaptive integration and get supporting points
    integral, supporting_points_x, supporting_points_y = sci_integration(gauss_pdf, a, b, num_support_points)
    print(f'Integral for bounds {a} and {b} is: {integral}')

    integral_lim = py_integration(func=gauss_pdf, int_l=a, int_u=b, n_points=num_support_points)
    print(f'Integral for bounds {a} and {b} for {num_support_points} is: {integral_lim}')

    print(f'Difference between SciPy and Own: {torch.abs(integral_lim - integral)}')

    if plot:
        # Plot the original function and supporting points
        x_values = np.linspace(a, b, 1000)
        y_values = gauss_pdf(x_values)
        plt.plot(x_values, y_values, label='Original Function')
        plt.scatter(supporting_points_x, supporting_points_y, color='red', label='Supporting Points')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Original Function and Supporting Points')
        plt.show()

