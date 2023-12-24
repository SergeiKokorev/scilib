import os
import sys

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from linalg import arnoldi, conj



if __name__ == "__main__":

    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    b = [1, 2, 3]

    a_conj = conj(A)

    Q, h = arnoldi(A, b, 1)

    print('Q')
    for q in Q:
        print(q)

    print('h')
    for hi in h:
        print(hi)
