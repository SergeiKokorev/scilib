import os
import sys

import matplotlib.pyplot as plt
import numpy as np


from scipy.optimize import curve_fit

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from optimize import lm, gauss_newton


# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c

def func(x, b1, b2):
    return b1 * x / (b2 + x)


if __name__ == "__main__":

    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3)
    rng = np.random.default_rng()
    y_noise = 0.2 * rng.normal(size=xdata.size)
    ydata = y + y_noise

    popt, pcov = curve_fit(func, xdata, ydata, p0=[7.0, 7.0])
    popt_lma = lm(func, xdata, ydata, p0=[7.0, 7.0])
    p0 = [1.0, 1.0]
    popt_gna = gauss_newton(func, xdata, ydata, p0)


    fig, ax = plt.subplots()
    ax.plot(xdata, ydata, 'b-', label='data')
    ax.plot(xdata, func(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
    ax.plot(xdata, func(xdata, *popt_lma), 'k-', label='fit LMA: a=%5.3f, b=%5.3f' % tuple(popt_lma))
    ax.plot(xdata, func(xdata, *popt_gna), 'y--', label='fit GNA: a=%5.3f, b=%5.3f' % tuple(popt_gna))

    ax.grid(True)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
