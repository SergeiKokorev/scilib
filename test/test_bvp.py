import os
import sys
import numpy as np
import matplotlib.pyplot as plt


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)


from integrate import solve_bvp


def tempeq(x, y):
    t1 = y[1]
    t0 = y[0]
    return np.array([t1, 0.1 * (t0 - 20)])


def bcres(ya, yb):
    return np.array([ya[0] - 40, yb[0] - 200])


if __name__ == "__main__":
    
    x = np.linspace(0, 10, 5)
    xext = np.linspace(0, 10, 500)
    y = np.zeros((2, x.size))

    solve_bvp(tempeq, bcres, x, y)
