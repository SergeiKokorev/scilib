from math import floor
from linalg import zeros, solve


def __fact(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * __fact(n - 1)


def __difference_coefficient(m: int, n: int) -> list:

    '''
        Returns list of coefficient for finite difference for the m-th derivative with accuracy n
        -----------------------------------------------------------------------------------------
        Parameters:     m: int
                            Derivative order
                        n: int
                            Accuracy n = 2, 4, 6, 8
    '''

    p = int(2 * floor((m + 1) / 2) - 1 + n - 1) / 2
    num_points = int(2 * p + 1)

    pm = zeros((num_points, num_points))
    rhs = zeros(num_points)

    for i in range(num_points):
        for j in range(num_points):
            pm[i][j] = (-p + j) ** i
        if i == m:
            rhs[i] = __fact(m)
    
    return solve(pm, rhs)


def forward_derivative(f: callable, x0: float, h: float=1e-5, n: int=1, args: tuple=()) -> float:

    '''
        Compute derivative of the function f at the point x0 by using forward 
        finit-divided-difference
        ---------------------------------------------------------------------
            Parameters      f: callable:
                                A function whose derivative to be found
                            x0: float:
                                A points at which the nth derivative is found
                            h: float, optional
                                An argument step of the function. f(x + h * x).
                                Default 1e-5
                            n: int, optional
                                Order of the derivative. Default 1
                            args: tuple, optional
                                Extra arguments to pass to the function.
                                Default no extra arguments passes to the function
    '''

    return (f(x0 + x0 * h, *args) - f(x0, *args)) / (h * x0)


def backward_derivative(f: callable, x0: float, h: float=1e-5, n: int=1, args: tuple=()) -> float:

    '''
        Compute derivative of the function f at the point x0 by using backward 
        finit-divided-difference
        ----------------------------------------------------------------------
            Parameters      f: callable:
                                A function whose derivative to be found
                            x0: float:
                                A points at which the nth derivative is found
                            h: float, optional
                                An argument step of the function. f(x + h * x).
                                Default 1e-5
                            n: int, optional
                                Order of the derivative. Default 1
                            args: tuple, optional
                                Extra arguments to pass to the function.
                                Default no extra arguments passes to the function
    '''

    return (f(x0, *args) - f(x0 - x0 * h, *args)) / (h * x0)


def derivative(f: callable, x0: float, h: float=1e-5, m: int=1, n: int=2, args: tuple=()) -> float:

    '''
        Compute derivative of the function f at the point x0 by using mid 
        finit-divided-difference
        -----------------------------------------------------------------
            Parameters      f: callable:
                                A function whose derivative to be found
                            x0: float:
                                A points at which the nth derivative is found
                            h: float, optional
                                A uniform grid spacing between each finite difference interval,
                                and xn = x0 + n * h. Default 1e-5
                            m: int, optional
                                Order of the derivative. Default 1
                            n: int, optional
                                Computation accuracy O(h^n).
                            args: tuple, optional
                                Extra arguments to pass to the function.
                                Default no extra arguments passes to the function
    '''
    if n % 2 != 0:
        raise RuntimeError('The accuracy must be even number. The accuracy O(h^n). Where n must be 2, 4, 6 etc.')

    coef = __difference_coefficient(m, n)
    num_points = floor((2 * floor((m + 1) / 2) - 1 + n) / 2)
    return sum([c * f(x0 + h * (i - num_points), *args) for i, c in enumerate(coef)]) / (h ** m)
