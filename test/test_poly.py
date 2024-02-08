import os
import sys
import numpy as np


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)

from poly import Polynomial



if __name__ == "__main__":
    
    print('Addition')
    p1 = Polynomial([2, 3, 5, 7])
    p2 = Polynomial([2, 0, 1])
    print(f'{p1 = }')
    print(f'{p2 = }')
    print(f'{p1 + p2 = }')
    p1 += p2
    print(f'p1 += p2: {p1}')
    print('Call')
    print(f'{p1(2) = }')
    print('Substraction')
    print(f'{p1 - p2 = }')
    print(f'{p2 - p1 = }')
    p1 -= p2
    print(f'p1 -= p2: {p1}')

    print('Multiplication')
    p1 = Polynomial([5, 25, 0, 1])
    p2 = Polynomial([1, -3, 0])
    print(f'{p1 = }')
    print(f'{p2 = }')
    print(f'{p1 * p2 = }')
    print(f'{p2 * p1 = }')
    p1 *= p2
    print(f'p1 *= p2: {p1}')
    for i in range(7):
        print(f'Order {i} {p1.polyder(i)}')

    print(f'{p1 = }')
    print(str([np.round(p, 4) for p in p1.roots()])[1:-1])
    print(np.roots(p1.coef))

    pt = Polynomial([1e-5, 2, 3, 5])
    print(f'{pt = }')
    print(f'{pt.trim(1e-4) = }')

    print(f'{p1 = }')
    print(f'{p1.cutdeg(2) = }')
