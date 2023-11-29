import os
import sys
import numpy as np
import matplotlib.pyplot as plt

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from integrate import solve_ipv


def mass_spring_damp(t, x, u, w, ksi):
    return [
        x[1],
        u - 2 * w * ksi * x[1] - w * x[0]
    ]


if __name__ == "__main__":

    m = 20
    k = 2
    c = 4
    g = 9.81

    w = (k / m) ** 0.5
    ksi = c / (2 * m * w)
    args = (g, w, ksi)
    
    tspan = (0, 60)

    tsol, xsol = solve_ipv(mass_spring_damp, tspan, (0, 0), args=args)
    
    fig, ax = plt.subplots()
    ax.plot(tsol, xsol[0], color='b', label='Position')
    ax.plot(tsol, xsol[1], color='r', label='Velocity')

    ax.grid(True)
    fig.legend()

    plt.show()
