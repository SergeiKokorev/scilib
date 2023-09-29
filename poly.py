from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from optimize import root


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


class __AbstractPoly(ABC):

    def __init__(self, coef: list, domain=None, window=None, symbol='x'):
        self.coef = coef
        self.sym = symbol
        self.domain = [-1, 1] if not domain else domain
        self.window = window
        super().__init__()

    @abstractproperty
    def degree(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Polynomial(__AbstractPoly):

    def __init__(self, coef: list, domain=None, window=None, symbol='x'):
        super().__init__(coef, domain, window, symbol)

    @property
    def degree(self):
        return len(self.coef) - 1

    def __add__(self, p:Polynomial):

        if not isinstance(p, Polynomial): 
            raise TypeError(f'unsupported operand type(s) for +: {type(p)} and {self.__class__.__name__}')

        c1 = [c1i for c1i in self.coef]
        c2 = [c2i for c2i in p.coef]
        if self.degree > p.degree:
            for i in range(self.degree - p.degree):
                c2.insert(0, 0.0)
        elif self.degree < p.degree:
            for i in range(p.degree - self.degree):
                c1.insert(0, 0.0)
        
        return Polynomial([c1i + c2i for c1i, c2i in zip(c1, c2)])
    
    def __radd__(self, p: Polynomial):
        return self.__add__(p)
    
    def __iadd__(self, p: Polynomial):
        return self.__add__(p)

    def __sub__(self, p:Polynomial):

        if not isinstance(p, Polynomial): 
            raise TypeError(f'unsupported operand type(s) for +: {type(p)} and {self.__class__.__name__}')

        c1 = [c1i for c1i in self.coef]
        c2 = [c2i for c2i in p.coef]
        if self.degree > p.degree:
            for i in range(self.degree - p.degree):
                c2.insert(0, 0.0)
        elif self.degree < p.degree:
            for i in range(p.degree - self.degree):
                c1.insert(0, 0.0)
        
        return Polynomial([c1i - c2i for c1i, c2i in zip(c1, c2)])
    
    def __rsub__(self, p: Polynomial):
        
        if not isinstance(p, Polynomial): 
            raise TypeError(f'unsupported operand type(s) for +: {type(p)} and {self.__class__.__name__}')

        c1 = [c1i for c1i in self.coef]
        c2 = [c2i for c2i in p.coef]
        if self.degree > p.degree:
            for i in range(self.degree - p.degree):
                c2.insert(0, 0.0)
        elif self.degree < p.degree:
            for i in range(p.degree - self.degree):
                c1.insert(0, 0.0)
        
        return Polynomial([c2i - c1i for c1i, c2i in zip(c1, c2)])
    
    def __isub__(self, p: Polynomial):
        return self.__sub__(p)

    def __mul__(self, p: Polynomial):
 
        if not isinstance(p, Polynomial): 
            raise TypeError(f'unsupported operand type(s) for +: {type(p)} and {self.__class__.__name__}')
 
        degree = self.degree + p.degree
        prod = [0.0 for i in range(degree + 1)]

        for i in range(self.degree + 1):
            for j in range(p.degree + 1 ):
                prod[i + j] += self.coef[i] * p.coef[j]

        return Polynomial(prod)

    def __rmul__(self, p: Polynomial):
        return self.__mul__(p)

    def __imul__(self, p: Polynomial):
        return self.__mul__(p)

    def __call__(self, x):
        return sum([ci * x ** (self.degree - i) for i, ci in enumerate(self.coef)])

    def __str__(self):
        s = ''
        for i in range(self.degree + 1):
            sign = '-' if self.coef[i] < 0 else '+'
            if self.coef[i]:
                if i == 0:
                    s += f'{self.coef[i]} {self.sym}**{self.degree}'
                elif self.degree - i == 1:
                    s += f' {sign} {abs(self.coef[i])} {self.sym}'
                elif self.degree - i == 0:
                    s += f' {sign} {abs(self.coef[i])}'
                else:
                    s += f' {sign} {abs(self.coef[i])} {self.sym}**{self.degree - i}'
        return s

    def __repr__(self):
        s = ''
        for i in range(self.degree + 1):
            sign = '-' if self.coef[i] < 0 else '+'
            if self.coef[i]:
                if i == 0:
                    s += f'{self.coef[i]} {self.sym}**{self.degree}'
                elif self.degree - i == 1:
                    s += f' {sign} {abs(self.coef[i])} {self.sym}'
                elif self.degree - i == 0:
                    s += f' {sign} {abs(self.coef[i])}'
                else:
                    s += f' {sign} {abs(self.coef[i])} {self.sym}**{self.degree - i}'
        return s
    
    def copy(self):
        return Polynomial(self.coef)

    def polyder(self, n: int) -> Polynomial:
        return Polynomial(polyder(self.coef, n))
    
    def roots(self):
        return root(self.coef)
    
    def trim(self, tol=0):
        return Polynomial(self.coef[1:] if self.coef[0] <= tol else self.coef)
    
    def cutdeg(self, deg):
        if deg > self.degree:
            return self.copy()
        else:
            return Polynomial(self.coef[self.degree - deg:])


