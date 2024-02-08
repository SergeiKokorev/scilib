import os
import sys

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from linalg import gsa


if __name__ == "__main__":

    s = [[3, 2, 2], [1, 2, 1], [2, 2, 3]]
    u = gma(s)

    print(u)
