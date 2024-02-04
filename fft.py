from __future__ import annotations

import math

from typing import List


class ComplexNumber:

    def __init__(self, real, imag=0):
        self.__real = real
        self.__imag = imag

    @property
    def real(self):
        return self.__real
    
    @property
    def imag(self):
        return self.__imag

    @property
    def r(self) -> float:
        return (self.real ** 2 + self.imag ** 2) ** 0.5

    @property
    def teta(self) -> float:
        return math.atan2(self.imag, self.real)

    @classmethod
    def exp(cls, teta: float) -> ComplexNumber:
        # print(f'exp(teta) {teta = } = ({math.cos(teta) = }, {math.sin(teta) = })')
        # print(f'{round(ComplexNumber(math.cos(teta), math.sin(teta)), 3) = }')
        return ComplexNumber(math.cos(teta), math.sin(teta))

    def conjugate(self):
        return ComplexNumber(self.real, (-1) * self.imag)

    def __transform__(self, other) -> ComplexNumber:

        if not isinstance(other, float | int | ComplexNumber | complex):
            raise TypeError(f"unsupported operand type(s) for '+': {type(self)} and {type(other)}")
        elif isinstance(other, int | float):
            other = ComplexNumber(other)
        elif isinstance(other, complex):
            other = ComplexNumber(other.real, other.imag)
        
        return other

    def __abs__(self) -> float:
        return (self.real ** 2 + self.imag ** 2) ** 0.5

    def __add__(self, other) -> ComplexNumber:
        other = self.__transform__(other)
        return ComplexNumber(self.real + other.real, self.imag + other.imag)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self.__transform__(other)
        return ComplexNumber(self.real - other.real, self.imag - other.imag)
    
    def __rsub__(self, other):
        other = self.__transform__(other)
        return ComplexNumber(other.real - self.real, other.imag - self.imag)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        other = self.__transform__(other)

        return ComplexNumber(self.real * other.real - self.imag * other.imag,
        self.real * other.imag + self.imag * other.real)
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self.__transform__(other)
        devider = other.real ** 2 + other.imag ** 2
        return ComplexNumber(
            (self.real * other.real + self.imag * other.imag) / divider,
            (self.imag * other.real - self.real * other.imag) / divider
        )
    
    def __rtruediv__(self, other):
        other = self.__transform__(other)

        divider = other.real ** 2 + self.real ** 2
        return ComplexNumber(
            (self.real * other.real + self.imag * other.imag) / divider,
            (self.real * other.imag - self.imag * other.real) / divider
        )

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __round__(self, ndigits=0):
        if self.imag >= 0:
            return f"({round(self.real, ndigits)} + {round(self.imag, ndigits)}j)"
        else:
            return f"({round(self.real, ndigits)} - {abs(round(self.imag, ndigits))}j)"

    def __str__(self):
        if self.imag >= 0:
            return f"({self.real} + {self.imag}j)"
        else:
            return f"({self.real} - {abs(self.imag)}j)"

def fft1(xin: List(float | complex | ComplexNumber)) -> List(ComplexNumber):

    '''
    Performs the Fast Fourier Transform (FFT) for a given signal.
    =============================================================

    Parameters:
    -----------

    xin:     array_like
        List of complex numbers. The input signal to be transformed.
    
    Returns:
    --------

    tr_signal:  array_like.
        The list of complex numbers. The transformed signal.
    
    '''

    n = len(xin)
    
    if n % 2 != 0:
        raise RuntimeError(f'Algorithm fft1 assumes N is a power of 2')

    xout = [0.0 for i in range(n)]

    for k in range(n//2):

        even = sum([xi * ComplexNumber.exp(-2 * math.pi * k * m / (n // 2))
        for m, xi in enumerate(xin[0::2])])
        odd = sum([xi * ComplexNumber.exp(-2 * math.pi * k * m / (n // 2))
        for m, xi in enumerate(xin[1::2])])

        rate = ComplexNumber.exp(-2 * math.pi * k / n)

        xout[k] = even + rate * odd
        xout[k + n//2] = even - rate * odd

    return xout
