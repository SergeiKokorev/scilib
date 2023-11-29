import os
import sys

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)
from linalg import eig, normalize


if __name__ == "__main__":

    a = [
        [10, -8, 7],
        [2, 4, 3],
        [1.2, 18, -4]
    ]


    res = eig(a)

    print(res[0])
    for r in res[1]:
        print(normalize(r))

