import os
import sys
import matplotlib.pyplot as plt
import numpy as np

DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir
))
sys.path.append(DIR)

from poly import CubicSpline



if __name__ == "__main__":

    knots = ((0, 2), (1, 3), (2, -1), (3, -4), (4, 4))
    # knots = ((0, 1), (1, 3), (2, 2))
    # knots = ((-1, 0.5), (0, 0), (3, 3))
    # knots = [[0, 21], [1, 24], [2 ,24], [3, 18], [4, 16]]
    spline = CubicSpline(knots)
    
    points = []
    intervals = []
    coefficients = []
    point = []
    for p in spline.poly:
        points.extend(p.compute())
        intervals.extend(p.interval)
        coefficients.append(p.coefficients)
        point.append(p(p.interval[0]))

    points = np.array(points)
    knots = np.array(knots)

    fig, ax = plt.subplots()
    ax.scatter(knots[:,0], knots[:,1], marker='o', color='r')
    ax.plot(points[:,0], points[:,1], marker='', color='b')

    ax.grid()
    
    plt.show()
