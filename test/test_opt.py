import os
import sys
import math


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)


import optimize as opt


def f1(x):
    return (x - 3) * (x - 1) * (x - 1)


def f2(x):
    return [x[0] * math.cos(x[1]) - 4,
            x[1] * x[0] - x[1] - 5]


def f3(x):
    return [x[0] ** 2 + x[1] * x[0] - 10 * x[2],
        x[1] + 3 * x[0] * x[1] ** 2 - 57 * x[2],
        x[0] ** 3 + x[1] * x[0] - 2 * x[2] - 12]


def poly(x, args):
    n = len(args) - 1
    return sum([ai * x ** (n - i) for i, ai in enumerate(args)])


def pnprime(x, args, n):

    m = len(args) - 1        
    if n == 1:
        print([ai * (m - i) for i, ai in enumerate(args[:-1])])
        return sum([ai * (m - i) * x ** (m - i - 1) for i, ai in enumerate(args[:-1], 1)])
    elif n == 0:
        return poly(x, args)
    else:
        return pnprime(x, [ai * (m - i) for i, ai in enumerate(args[:-1])], n - 1)


if __name__ == "__main__":

    x = opt.newton_raphson(f1, 0.0)
    print(x)

    root = opt.fsolve(f2, [1.0, 1.0])
    print([round(yi, 6) for yi in root])
    print([round(yi, 4) for yi in f2(root)])
    root = opt.fsolve(f3, [1.0, 1.0, 1.0])
    print([round(yi, 6) for yi in root])
    print([round(yi, 4) for yi in f3(root)])

    args = [1, -9, 26, -24]
    root = opt.root(args)
    print(root)
    for r in root:
        print(poly(r, args))

    args = [1, -7, 15, -9] # Multiple root x = 3, (x - 1)(x - 3)(x - 3)
    root = opt.root(args)
    print(root)
    for r in root:
        print(poly(r, args))

    args = [1, 2, 3, 4] # Complex roots
    root = opt.root(args)
    print(root)
    for r in root:
        print(r, poly(r, args))
