import os
import sys

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)


from utils import *


if __name__ == "__main__":

    mat = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    
    for m in swap_rows(mat, 0, 2, copy=False):
        print(m)

    print()

    for m in swap_cols(mat, 0, 2, copy=False):
        print(m)
    swap_rows(mat, 0, 2)
    print(mat)

    print(max_element(mat, 1, 1))
