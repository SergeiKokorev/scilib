import os
import sys

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from linalg import gaussian, mult, backward_sub



if __name__ == "__main__":

    A = [
        [5, 2, 4],
        [1, 0, .3],
        [2., 1, 5]
    ]
    b = [2, 4, 3]

    an, bn = gaussian(A, b)
    x = backward_sub(an, bn)

    print('A')
    for ai in an:
        print(ai)
    print(f'{bn = }')
    print(f'{x = }')
    print(f'Checking: {mult(A, x)}')
