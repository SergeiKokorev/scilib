import os
import sys

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from optimize import bisect


def func(x):
    return x ** 3 - x - 2


if __name__ == "__main__":

    xsol = bisect(func, 1, 2, tol=1e-4)
    print(f'{xsol = }')
    print(f'{func(xsol) = }')

