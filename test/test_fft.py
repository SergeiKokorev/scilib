import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

import time

CDIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir
))
sys.path.append(CDIR)

from fft import FFT  as fft
from windows import *


def fun(x):
    return 2 * x ** 3 + 3 * x - 7


if __name__ == "__main__":

    fs = 1000                       # Sampling frequency
    per = 1 / fs                    # Sampling period
    n = 1500                        # Length of signal
    t = per * np.arange(n)          # Time vector

    signal = 0.8 + 0.7 * np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
    # signal = fun(np.random.normal())
    x = np.array(signal) + 2 * np.random.randn(t.size)
    n = len(x)

    w = sp.signal.windows.bartlett(x.size)
    xws = x * w

    xsp = sp.fft.fft(x)
    xspw = sp.fft.fft(xws)
 
    x_fft = fft(x)
    w = bartlett(n)
    xw = [xi * wi for xi, wi in zip(x, w)]
    xw_fft = fft(xw)

    amp = x_fft.magnitude
    amp_xw = xw_fft.magnitude

    freq = fs / n * np.arange(n)
    ts = 1000 * t

    fig, axs = plt.subplots(4, 1)
    
    axs[0].set_title('Signal corrupted with zero-mean random noise')
    axs[0].plot(ts, x)

    axs[1].set_title('Windowed signal')
    axs[1].set_xlabel('t, [ms]')
    axs[1].set_ylabel('x(t)')
    axs[1].plot(ts, xws, label='Windowed scipy')
    axs[1].plot(ts, xw, label='Windowed scilib')
    axs[1].legend()

    axs[2].set_title('Complex magnitude of FFT spectrum scipy')
    axs[2].plot(freq, abs(xsp), label='Non windowed')
    axs[2].plot(freq, abs(xspw), label='Windowed')
    axs[2].set_xlabel('f, [Hz]')
    axs[2].set_ylabel('|fft(x)|')
    axs[2].legend()

    axs[3].set_title('Complex magnitude of FFT spectrum scilib')
    axs[3].plot(freq, amp, label='Non windowed')
    axs[3].plot(freq, amp_xw, label='Windowed')
    axs[3].set_xlabel('f, [Hz]')
    axs[3].set_ylabel('|fft(x)|')
    axs[3].legend()    

    plt.show()
