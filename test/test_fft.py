import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt

# from numpy.fft import fft
from scipy.fft import fft, fftfreq, fftshift


DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from fft import FFT as fft1

SAMPLE_RATE = 44100
DURATION = 5


def generate_sine_wave(freq, sample_rate, duration):

    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    return x, y


if __name__ == "__main__":

    N = 600
    T = 1.0 / 800.0
    x = np.linspace(0.0, N * T, endpoint=False)
    y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)

    yf = fft(y)
    ym = fft1(y)
    
    for k, (y1, y2) in enumerate(zip(yf, ym.frequency)):
        print(f'{k}\t:\t{np.round(y1, 4)}\t:\t{round(y2, 4)}')

    signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5])
    signal = np.arange(256)

    sp = fft(np.sin(signal))
    freq = fftfreq(signal.shape[-1])

    sp1 = fft1(np.sin(signal))
    freq1 = sp1.freq(signal.shape[-1])

    plt.plot(freq, sp.real, freq, sp.imag)
    plt.show()
    plt.plot(freq1, sp1.real, freq1, sp1.imag)
    plt.show()

    signal = np.array([-2, 8, 6, 4, 1, 0, 3, 5, 8], dtype=float)
    fourier = fft(signal)
    n = signal.size
    timestep = 0.1
    freq = fftfreq(n, d=timestep)
    print(freq)

    freq_my = fft1.freq(n, timestep)
    print(freq_my)

    freqs = fftfreq(10, 0.1)
    freqs_my = fft1.freq(10, 0.1)
    print(np.round(freqs, 3))
    print([round(f, 3) for f in freqs_my])
    print(np.round(fftshift(freqs), 3))
    print([round(f, 3) for f in fft1.shift(freqs_my)])
