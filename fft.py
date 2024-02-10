from __future__ import annotations

import math

from typing import List
from copy import deepcopy


def fft(signal: List(float | complex)) -> List[complex | float]:

    '''
    Performs the Fast Fourier Transform (FFT) for a given signal.
    =============================================================

    Parameters:
    -----------

    signal:     array_like
        List of complex numbers. The input signal to be transformed.
    
    Returns:
    --------

    tr_signal:  array_like.
        The list of complex numbers. The transformed signal.
    
    '''

    n = len(signal)
    
    if n % 2 != 0:
        raise RuntimeError(f'Algorithm fft assumes N is a power of 2')

    xout = [0.0 for i in range(n)]

    for k in range(n//2):

        even = sum([xi * Complex.exp(-2 * math.pi * k * m / (n // 2))
                    for m, xi in enumerate(signal[0::2])])
        odd = sum([xi * Complex.exp(-2 * math.pi * k * m / (n // 2))
                    for m, xi in enumerate(signal[1::2])])

        rate = Complex.exp(-2 * math.pi * k / n)

        xout[k] = even + rate * odd
        xout[k + n//2] = even - rate * odd

    return xout


class FFT:

    def __init__(self, signal, window='hanning', copy=False) -> None:
        self.__signal = deepcopy(signal) if copy else signal
        self.__frequency = fft(self.__signal)

    @property 
    def signal(self):
        return self.__signal

    @property
    def frequency(self) -> List[Complex | complex | float]:
        return self.__frequency

    @property
    def real(self) -> List[float]:
        return [f.real for f in self.__frequency]

    @property
    def imag(self) -> List[float]:
        return [f.imag for f in self.__frequency]

    @property
    def phase(self):
        return [math.atan2(f.imag, f.real) for f in self.frequency]
    
    @property
    def magnitude(self):
        return [abs(f) for f in self.__frequency]

    @property
    def density(self):
        return [
            d ** 2 if i == 0 else 2 * d ** 2 
            for i, d in enumerate(self.magnitude)
        ]

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

    def __round__(self, ndigits: int):
        return [round(f, ndigits) for f in self.__frequency]

    def __str__(self):
        out = '['
        for i, f in enumerate(self.__frequency):
            if i != len(self.__frequency) - 1:
                out += f'{f}\n'
            else:
                out += f'{f}]'


class Complex(complex):

    @classmethod
    def exp(cls, teta: float | complex) -> complex:
        return Complex(math.cos(teta), math.sin(teta))

    @property
    def teta(self):
        return math.atan2(self.imag, self.real)

    def __round__(self, ndigits: int):

        if not isinstance(ndigits, int):
            raise TypeError(f'Number of digits must be integer. Given {type(ndigits)}.')

        return Complex(round(self.real, ndigits), round(self.imag, ndigits))
