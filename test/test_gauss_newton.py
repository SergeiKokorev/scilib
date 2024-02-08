import os
import sys

import numpy as np
import matplotlib.pyplot as plt

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from optimize import gauss_newton


def jacobian(x, b1, b2):
    return [
        [-xi / (b2 + xi), b1 * xi / (b2 + xi) ** 2] for xi in x
    ]


def enzyme_mediate(s, vmax, km):
    '''
        Reaction rate in enzyme-mediate reaction
        rate = Vmax * [S] / (KM + [S])
    '''
    return vmax * s / (km + s)



if __name__ == "__main__":

    s = np.array([0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.74])
    rate = np.array([0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317])
    x = np.linspace(0, 4, 100)

    pinit = [0.9, 0.2]

    jac = jacobian(s, *pinit)
    popt = gauss_newton(enzyme_mediate, s, rate, pinit)
    print(f'Vmax = {round(popt[0], 4)}; KM = {round(popt[1], 4)}')


    ysol = enzyme_mediate(x, *popt)

    fig, ax = plt.subplots()
    ax.scatter(s, rate, marker='o', color='r', label='Measured data')
    ax.plot(x, ysol, marker='', color='k', label='Model Function')
    ax.grid()
    fig.legend()

    plt.show()
