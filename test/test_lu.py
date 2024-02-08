import os
import sys
import numpy as np
import scipy as sc

from numpy import linalg
from scipy import linalg as sla


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from linalg import eig, lu, inverse as inv, transpose, mult


if __name__ == "__main__":

    a = np.array([
        [12, -51, 4, 8],
        [6, 167, -68, -12],
        [-4, 24, -41, 34],
        [12, 16, 8, -45]
    ], dtype=float)

    a = [
            [-144, -51, 4, 2], 
            [6, 10, -68, 8.0], 
            [-4, 24, -197, 17],
            [-5, -6, -8.0, 12]
        ]

    print('Initial matrx')
    for ai in a:
        print(ai)
    print()

    P, L, U = sla.lu(a)

    print(f'{U = }', end='\n\n')
    print(f'{L = }', end='\n\n')
    print(f'{P = }', end='\n\n')

    print('A')
    print(P @ L @ U)

    p, l, u = lu(a)

    print('\nU')
    for ui in u:
        print([round(uij, 6) for uij in ui])    

    print('\nL')
    for li in l:
        print([round(lij, 6) for lij in li])

    print('\nP')
    for pi in p:
        print(pi)

    print()

    for ai in mult(p, mult(l, u)):
        print([round(aij, 5) for aij in ai])
