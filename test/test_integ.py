import os
import sys
import math
import matplotlib.pyplot as plt


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)

from integrate import *


def f1(t, y):
    return [
        math.sin(t) + math.cos(y[0]) + math.sin(y[1]),
        math.cos(t) + math.sin(y[1])
    ]


def f2(t, y):
    return -0.5 * y


if __name__ == "__main__":
    


    tsol, ysol = solve_ipv(f1, tspan=(0, 20), y0=[-1 , 1])

    fig, ax = plt.subplots()
    ax.plot(tsol, ysol[0], marker='', color='b')
    ax.plot(tsol, ysol[1], marker='', color='r')

    plt.show()
