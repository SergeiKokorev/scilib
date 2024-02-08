import os
import sys

from numpy.linalg import eig
from scipy import linalg

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from linalg import qr, inverse, transpose, mult, transpose




if __name__ == "__main__":

    # a1 = [
    #     [7, 9, 4, 3],
    #     [8, 1, 6, 5],
    #     [9, 2, 3, 6],
    #     [1, 8, 7, 3]
    # ]

    a1 = [
        [12, -51, 4],
        [6, 167, -68],
        [-4, 24, -41]
    ]


    print('Initial matrix A')
    for ai in a1:
        print([round(aij, 4) for aij in ai])

    v = eig(a1)
    print('Eigen vectors: ', v)

    Q, R = linalg.qr(a1)

    print('Q')
    print(Q)
    print('R')
    print(R)

    q, r = qr(a1)
    qt = transpose(q, copy=True)
    print(f'lam3 = {r[2][2] * q[2][2]}')

    print('\nQ')
    for qi in q:
        print([round(qij, 4) for qij in qi])

    print('\nChecking Q: Q^T = Q^-1')
    print('\nTranspose Q')
    for qi in transpose(q, copy=True):
        print([round(qij, 4) for qij in qi])
    print('\nInverse Q')
    for qi in inverse(q):
        print([round(qij, 4) for qij in qi])

    print('\nR')
    for ri in r:
        print([round(rij, 4) for rij in ri])

    print('\nChecking: A = QR')
    for m in mult(q, r):
        print([round(mi, 4) for mi in m])
