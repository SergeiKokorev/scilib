import os
import sys


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)

from linalg import *
from utils import *


if __name__ == "__main__":

    matrix1 = [
        [10, 12, 14],
        [1, 2.5, 3.0],
        [23, 12, 4.0]
    ]

    matrix2 = [
        [1.2, 2.5, 3.5],
        [0.1, 0.2, 2.5],
        [11, 1.1, 1.2]
    ]

    print('initial matrix')
    for m in matrix1:
        print(m)

    transpose(matrix1)
    print('After transpose')
    for m in matrix1:
        print(m)

    vector = [1.2, 2.5, 1.0, 4.5]
    print('initial vector ', vector)
    print('after transpose ', transpose(vector))

    for m in mult(matrix1, vector):
        print(m)

    print()

    for m in mult(matrix1, matrix2):
        print(m)

    print(mult(1.5, vector))
    print(mult(vector, 1.5))
    print(mult(1.5, matrix1))
    print(mult(matrix1, 1.5))
    print(mult(1.5, 1.5))

    matrix1 = [
        [10, 12, 14],
        [1, 2.5, 3.0],
        [23, 12, 4.0]
    ]

    for m in inverse(matrix1):
        print(m)

    matrix1 = [
        [10, 12, 14],
        [1, 2.5, 3.0],
        [23, 12, 4.0]
    ]

    print(det(matrix1), end='\n\n')

    a = [[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]]

    p, l, u = lu(a)
    n = 5
    mat_pow = pow(a, n)
    print(f'Matrix a to power {n}')
    for pi in mat_pow:
        print(pi)

    print('P')
    for pi in p:
        print(pi)
    print('L')
    for li in l:
        print([round(lij, 4) for lij in li])
    print('U')
    for ui in u:
        print([round(uij, 4) for uij in ui])
    print()
    for ai in mult(p, mult(l, u)):
        print([round(aij, 1) for aij in ai])

    print('Inverse U')
    for ui in inverse(u):
        print([round(uij, 4) for uij in ui])

    n = len(u)
    lam = diag(u)
    tu = sub(u, lam)
    lam_inv = deepcopy(lam)
    for i in range(n):
        lam_inv[i][i] = 1 / lam[i][i]
    
    uinv = identity(n)
    for i in range(1, n):
        uinv = add(pow(mult(mult((-1), lam_inv), tu), i), uinv)
    uinv = mult(uinv, lam_inv)

    print('inverse U by using backward substitution')
    for ui in uinv:
        print([round(uij, 4) for uij in ui])

    a = [[3, 2, 0], [1, -1, 0], [0, 5, 1]]
    b = [2, 4, -1]

    x = solve(a, b)
    print('Solution')
    print([round(xi, 3) for xi in x])
    print(mult(a, x))
