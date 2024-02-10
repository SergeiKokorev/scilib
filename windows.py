from math import pi, cos

def hamming(n):
    return [0.54 - 0.46 * cos(2 * pi * j / (n - 1)) for j in range(n)]


def hann(n):
    return [0.5 - 0.5 * cos(2 * pi * j / (n - 1)) for j in range(n)]


def bartlett(n):
    return [2 / (n - 1) * ((n - 1) / 2 - abs(j - (n - 1) / 2)) for j in range(n)]


def blackman(n):
    return [0.42 - 0.5 * cos(2 * pi * j / n) + 0.08 * cos(4 * pi * j / n) 
    for j in range(n)]
