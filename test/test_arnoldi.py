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
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

    b = [1, 2, 3]

    Q, H = arnoldi(A, b, 2)

    print('Q')
    for q in Q:
        print([round(qi, 2) for qi in q])

    print('H')
    for h in H:
        print([round(hi, 2) for hi in h])

    print('Dot')
    print(dot(Q, H))
    
