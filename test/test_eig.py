import os
import sys

import numpy as np

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)
from linalg import eig, norm, transpose


if __name__ == "__main__":

    A = [
        [1, 2, 3, 7, 3],
        [4, 5, 6, 8, 5],
        [7, 8, 9, 12, 7],
        [12, 11, 8, 7, 10],
        [0.2, 0.1, 0.3, 0.5, 11]
    ]

    eigval, eigvec = np.linalg.eig(A)
    eigvec = transpose(eigvec)

    print('\nnumpy')
    for va, ve in zip(eigval, eigvec):
        print(f'{round(va, 3)} : {[round(vi, 3) for vi in ve]} : {norm(ve)}')

    eigval, eigvec = eig(A)

    print('\nmy')
    for val, vec in zip(eigval, transpose(eigvec)):
        print(f'{round(val, 3)} : {[round(vi, 3) for vi in vec]} : {norm(vec)}')
