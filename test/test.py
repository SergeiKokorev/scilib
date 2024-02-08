import os
import sys
import math


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)

from linalg import lu, inverse, transpose, solve



if __name__ == "__main__":

    a = [[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]]

    p, l, u = lu(a)

    print(p)
    print(inverse(p))
    print(transpose(p))

    a = [
        [1, 2, 0, 0, 0],
        [3, 4, 5, 0, 0],
        [0, 6, 7, 8, 0],
        [0, 0, 9, 10, 11],
        [0, 0, 0, 12, 13]
    ]
    b = [2, 3, 5, 4, 7]

    print(solve(a, b))
    print(solve(a, b, tridiagonal=True))
