import os
import sys


from math import floor


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(DIR)


import misc



def fact(n):
    if n == 0:
        return 1
    if n == 1:
        return 1
    else:
        return n * fact(n - 1)


def func(x, *args):
    return sum([a * x ** i for a, i in enumerate(args)])


def func1d(x, *args):
    return sum([a * i * x ** (i - 1) for a, i in enumerate(args)])


def p(m, n):
    return 2 * floor((m + 1) / 2) -1 + n


def ap(p, n):
    return [(-1) ** (pi + 1) * (fact(n) ** 2) / (fact(m) * pi * fact(n - pi) * fact(n + pi)) for pi in range(1, p - 1)]


if __name__ == "__main__":

    args = tuple([0.2, 0.1, 3.2, 0.25])
    x0 = 1.2
    m = 1
    n = 2
    print(f'Exact derivative = {(d1:=func1d(x0, *args))}')
    d = misc.derivative(func, x0, m=m, n=n, args=args)
    print(f'Approximate {m} derivative, accuracy {n} = {d}')
    print(f'Relative error {(abs(d1 - d) / d) * 100}%')