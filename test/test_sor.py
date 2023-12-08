import os
import sys

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from optimize import sor
from linalg import mult


if __name__ == "__main__":

    A = [
        [4, -1, -6, 0],
        [-5, -4, 10, 8],
        [0, 9, 4, -2],
        [1, 0, -7, 5]
    ]
    b = [2, 21, -12, -6]
    x0 = [0.0, 0.0, 0.0, 0.0]
    omega = 0.5

    xsol = sor(A, b, x0, omega)
    print(f'Solution : {[round(xi, 3) for xi in xsol]}')
    print(f'Check system {[round(xi, 6) for xi in mult(A, xsol)]}')

    A = [[2, 1], [1, 2]]
    b = [2, 1]
    x0 = [-0.5, 0.5]
    omega = 1

    xsol = sor(A, b, x0, omega)
    print(f'Solution : {[round(xi, 3) for xi in xsol]}')
    print(f'Check system {[round(xi, 6) for xi in mult(A, xsol)]}')
