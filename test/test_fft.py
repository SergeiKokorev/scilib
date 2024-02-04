import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as sp

# from numpy.fft import fft
from scipy.fft import fft


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from fft import fft1, ComplexNumber

SAMPLE_RATE = 44100
DURATION = 5


def generate_sine_wave(freq, sample_rate, duration):

    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    return x, y


if __name__ == "__main__":

    # x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
    # plt.plot(x,y)
    # plt.show()
    N = 600
    T = 1.0 / 800.0
    x = np.linspace(0.0, N * T, endpoint=False)
    y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)

    yf = fft(y)
    ym = fft1(y)
    
    for k, (y1, y2) in enumerate(zip(yf, ym)):
        print(f'{k}\t:\t{np.round(y1, 4)}\t:\t{round(y2, 4)}')
