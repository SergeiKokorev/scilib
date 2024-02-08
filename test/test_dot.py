import os
import sys
import numpy as np


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)

from linalg import dot


if __name__ == "__main__":

    a, b = 2, 3
    print(np.dot(a, b))
    print(dot(a, b))

    m = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    v = [7, 8]
    print(np.dot(v, m))
    print(dot(v, m))

    m = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    v = [7, 8, 9]
    print(np.dot(m, v))
    print(dot(m, v))

    print(np.dot(a, m))
    print(np.dot(m, a))
    print(dot(a, m))
    print(dot(m, a))

    m1 = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    m2 = [
        [7, 8],
        [9, 10],
        [11, 12]
    ]

    print('m1.dot(m2)')
    print(np.dot(m1, m2))
    for e in dot(m1, m2):
        print(e)
    print('m2.dot(m1)')
    print(np.dot(m2, m1))
    for e in dot(m2, m1):
        print(e)
