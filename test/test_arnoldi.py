import os
import sys


import numpy.linalg as linalg


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from linalg import arnoldi,dot, transpose, eig


if __name__ == "__main__":

    A = [
        [1, 2, 3, 7],
        [4, 5, 6, 8],
        [7, 8, 9, 12],
        [12, 11, 8, 7]
    ]

    b = [1, 2, 3, 7]

    Q, H = arnoldi(A, b, 4)

    print('Q')
    for q in Q:
        print([round(qi, 3) for qi in q])

    print('\nH')
    for h in H:
        print([round(hi, 3) for hi in h])

    print('\nCheck. A = QHQ*')
    for q in dot(Q, dot(H, transpose(Q))):
        print([round(qi, 3) for qi in q])
