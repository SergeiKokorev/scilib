import os
import sys

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from linalg import arnoldi, conj, dot


if __name__ == "__main__":

    A = [
        [1, 2, 3, 7],
        [4, 5, 6, 8],
        [7, 8, 9, 12],
        [12, 11, 8, 7]
    ]

    b = [1, 2, 3, 7]

    Q, H = arnoldi(A, b, 3)

    print('Q')
    for q in Q:
        print([round(qi, 3) for qi in q])

    print('\nH')
    for h in H:
        print([round(hi, 3) for hi in h])

    print('\nCheck')
    for a in dot(Q, dot(A, conj(Q))):
        print([round(ai, 3) for ai in a])

