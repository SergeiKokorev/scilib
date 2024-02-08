import os
import sys
import numpy as np


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)

from optimize import polyfit


def func(x, a, b, c):
    return a - x ** 2 + b * np.cos(2 * np.pi * x / c)


if __name__ == "__main__":

    pass
