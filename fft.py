from __future__ import annotations

import math

from typing import List


def fft1(xin: List(float | complex | ComplexNumber)) -> List[ComplexNumber | complex | float]:

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


class FFT:

    def __init__(self, signal: List[ComplexNumber | complex | float]):

        self.__signal = signal
        self.__frequency = fft1(self.signal)

    @property
    def signal(self):
        return self.__signal

    @signal.setter
    def signal(self, signal: List[ComplexNumber | complex | float]):
        if not hasattr(signal, '__iter__'):
            raise TypeError('Unsupported variable type for signal')
        elif not all([isinstance(s, complex | ComplexNumber | float | int) for s in signal]):
            raise TypeError('Unsupported variable type for signal. Signal must be List[float | int | ComplexNumber | complex]')
        self.__signal = signal
        self.__frequency = fft1(self.signal)

    @property
    def frequency(self):
        return self.__frequency
    
    @property
    def real(self):
        return [f.real for f in self.frequency]
    
    @property
    def imag(self):
        return [f.imag for f in self.frequency]
    
    @property
    def phase(self):
        return [math.atan2(cn.imag, cn.real) for cn in self.frequency]
    
    @property
    def amplitute(self):
        return [(cn.real ** 2 + cn.imag ** 2) ** 0.5 for cn in self.frequency]

    @classmethod
    def freq(cls, n: int, d: float = 1.0) -> List[float]:

        f = [0.0 for i in range(n)]
        if n % 2 == 0:
            for i in range(n):
                f[i] = i if i <= (n / 2 - 1) else (i - n)
                f[i] /= (d * n)
        else:
            for i in range(n):
                f[i] = i if i <= (n - 1) / 2 else i - n
                f[i] /= (d * n)

        return f
    
    @classmethod
    def shift(cls, freqs: List[float]) -> List[float]:

        n = len(freqs)
        fs = [0.0 for i in range(n)]
        if n % 2 == 0:
            for i, k in enumerate(range(int(n / 2), n)):
                fs[i] = freqs[k]
            for i in range(int(n / 2)):
                fs[int(n / 2 + i)] = freqs[i]
        else:
            for i, k in enumerate(range(int((n - 1) / 2 + 1), n)):
                fs[i] = freqs[k]
            for i in range(int((n - 1) / 2 + 1)):
                fs[int((n - 1) / 2 + i)] = freqs[i]
        return fs

    def __round__(self, ndigits=0):
        return [round(f, ndigits) for f in self.frequency]

    def __str__(self):
        out = '[ '
        for i, f in enumerate(self.frequency):
            if i % 2 == 0 and i != 0:
                out += f'\n{str(f)}\t'
            else:
                out += f'{str(f)}\t'
            out += ']'
        return out

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
            (self.real * other.real + self.imag * other.imag) / devider,
            (self.imag * other.real - self.real * other.imag) / devider
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
        return ComplexNumber(round(self.real, ndigits), round(self.imag, ndigits))

    def __str__(self):
        if self.imag >= 0:
            return f"{self.real} + {self.imag}j"
        else:
            return f"{self.real} - {abs(self.imag)}j"
