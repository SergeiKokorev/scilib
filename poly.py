from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from optimize import root
from linalg import solve


def _coef_gen(knots, degree=3):

    x = [k[0] for k in knots]
    y = [k[1] for k in knots]

    ''' ### Generate matrix A ### '''
    knots_num = len(knots)
    spline_degree = degree
    A = [[0 for i in range(4 * (knots_num - 1))] for j in range(4 * (knots_num - 1))]

    # Constrains Si(xi) = yi and Si+1(xi) = yi
    k = 0
    row = 0
    for i in range(knots_num - 1):
        d = spline_degree
        for j in range(k, k + spline_degree):
            A[i * 2][j] = x[i] ** d
            A[i * 2 + 1][j] = x[i + 1] ** d
            d -= 1
        A[i * 2][j + 1] = 1
        A[i * 2 + 1][j + 1] = 1
        k = k + spline_degree + 1
    
    # Constrains Si'(xi) = Si+1'(xi)
    row += (2 * (knots_num - 1))
    k = 0
    for i in range(row, row + knots_num - 2):
        d = spline_degree
        for j in range(k, k + spline_degree):
            A[i][j] = d * x[i - row + 1] ** (d - 1)
            A[i][j + spline_degree + 1] = (-1) * d * x[i - row + 1] ** (d - 1)
            d -= 1
        k = k + spline_degree + 1
        
    # Constrains Si''(xi) = Si+1''(xi)
    row += (knots_num - 2)
    k = 0
    for i in range(row, row + knots_num - 2):
        d = spline_degree
        for j in range(k, k + spline_degree - 1):
            A[i][j] = d * (d - 1) * x[i - row + 1] ** (d - 2)
            A[i][j + spline_degree + 1] = (-1) * d * (d - 1) * x[i - row + 1] ** (d - 2)
            d -= 1
        k = k + spline_degree + 1

    # Boundary natural spline S1''(x1) = 0 and Sn-1''(xn) = 0
    d = spline_degree
    for j in range(spline_degree - 1):
        A[-2][j] = d * (d - 1) * x[0] ** (d - 2)
        A[-1][j - spline_degree - 1] = d * (d - 1) * x[-1] ** (d - 2)
        d -= 1

    ''' ### Generate RHS ### '''
    rhs = [0.0 for i in range(4 * (knots_num - 1))]
    rhs[0] = y[0]

    col = 0
    for i, yi in enumerate(y[1:-1], 1):
        rhs[i + col] = yi
        rhs[i + 1 + col] = yi
        col += 1
    
    rhs[2 * (knots_num - 1) - 1] = y[-1]

    return solve(A, rhs)


def polyder(p:list, n:int) -> float:
    '''
        Computes polynomial derivative of n-order
        ---------------------------------------------------
        Parameters:      args : array_like
                            Polynomial coefficients a[0] * x**n + a[1] * x**(n - 1) + ... + a[n] = 0
                        n : int
                            Order of the derivative
        Returns:        result : float
                            The derivative of the specified order of a polynomial
    '''
    m = len(p) - 1        
    if n == 1:
        return [ai * (m - i) for i, ai in enumerate(p[:-1])]
    elif n == 0:
        return p
    else:
        return polyder([ai * (m - i) for i, ai in enumerate(p[:-1])], n - 1)


class Poly:

    def __init__(self, coef:list, symbol:str='x', interval:tuple=(0, 1)) -> None:
        self.__coef = coef
        self.__sym = symbol
        self.__interval = interval

    @property 
    def degree(self):
        return len(self.__coef) - 1
    
    @property
    def coefficients(self):
        return self.__coef
    
    @property
    def symbol(self):
        return self.__sym
    
    @coefficients.setter
    def coefficients(self, coef: list) -> None:
        
        if not hasattr(coef, '__iter__'):
            return None
        elif not all([isinstance(ci, (int, float)) for ci in coef]):
            return None
        
        self.__coef = coef

    @property
    def interval(self):
        return self.__interval
    
    @interval.setter
    def interval(self, interval: tuple):
        self.__interval = interval

    def __add__(self, p: Poly):
        return Poly(coef=[c1 + c2 for c1, c2 in zip(self.coefficients, p.coefficients)])
    
    def __radd__(self, p: Poly):
        return self.__add__(p)
    
    def __iadd__(self, p: Poly):
        return self.__add__
    
    def __sub__(self, p: Poly):

        if self.degree > p.degree:
            c1 = [ci for ci in self.coefficients]
            c2 = [0 if i > p.degree else (-1) * p.coefficients[p.degree - i] for i in range(self.degree, -1, -1)]
        else:
            c1 = [0 if i > self.degree else self.coefficients[self.degree - i] for i in range(p.degree, -1, -1)]
            c2 = [(-1) * c for c in p.coefficients]

        return Poly(coef=[c1i + c2i for c1i, c2i in zip(c1, c2)])

    def __rsub__(self, p: Poly):
        
        if self.degree > p.degree:
            c1 = [(-1) * ci for ci in self.coefficients]
            c2 = [0 if i > p.degree else  p.coefficients[p.degree - i] for i in range(self.degree, -1, -1)]
        else:
            c1 = [0 if i > self.degree else (-1) * self.coefficients[self.degree - i] for i in range(p.degree, -1, -1)]
            c2 = [c for c in p.coefficients]

        return Poly(coef=[c1i + c2i for c1i, c2i in zip(c1, c2)])

    def __isub__(self, p: Poly):
        return self.__sub__(p)

    def __mul__(self, p: Poly):
 
        if not isinstance(p, Poly): 
            raise TypeError(f'unsupported operand type(s) for +: {type(p)} and {self.__class__.__name__}')
 
        degree = self.degree + p.degree
        prod = [0.0 for i in range(degree + 1)]

        for i in range(self.degree + 1):
            for j in range(p.degree + 1 ):
                prod[i + j] += self.coefficients[i] * p.coefficients[j]

        return Poly(prod)

    def __rmul__(self, p: Poly):
        return self.__mul__(p)

    def __imul__(self, p: Poly):
        return self.__mul__(p)

    def __call__(self, x: float):
        return sum([ci * x ** (self.degree - i) for i, ci in enumerate(self.coefficients)])

    def __str__(self):
        d = self.degree
        res = ''

        for i, ai in enumerate(self.__coef):

            if not ai == 0:
                coef = str(abs(ai)) if not (abs(ai) == 1 and i != d) else ''
                n = f'^{d - i}' if not ((d - i) == 0 or (d - i) == 1) else ''
                s = self.__sym if not (d - i) == 0 else ''

                if i == d:
                    sign = ''
                elif ai < 0:
                    sign = ' - '
                else:
                    sign = ' + '

                res += coef + s + n + sign

        return res

    def polyder(self, n: int) -> Poly:
        return Poly(polyder(self.coefficients, n))

    def copy(self):
        return Poly(self.coefficients)

    def roots(self):
        return root(self.coefficients)
    
    def compute(self, x: tuple=None):
        if not x:
            delx = (self.__interval[1] - self.__interval[0]) / 10
            x = [self.__interval[0] + delx * i for i in range(11)]
        return [(xi, self(xi)) for xi in x]


class CubicSpline:

    def __init__(self, knots: list) -> None:
        self.__knots = knots
        self.__poly = []
        self.__coef = _coef_gen(knots)
        
        for i in range(0, 4 * (len(self.__knots) - 1), 4):
            self.__poly.append(Poly(self.__coef[i:i+4]))
        
        for k in range(len(self.__knots) - 1):
            self.__poly[k].interval = (self.__knots[k][0], self.__knots[k + 1][0])

    @property
    def knots(self):
        return self.__knots

    @property
    def poly(self):
        return self.__poly