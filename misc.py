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


def derivative(f: callable, x0: float, h: float=1e-5, n: int=1, args: tuple=()) -> float:

    '''
        Compute derivative of the function f at the point x0 by using mid 
        finit-divided-difference
        -----------------------------------------------------------------
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
    if n == 1:
        return (f(x0 + h * x0, *args) - f(x0 - h * x0, *args)) / (2 * h * x0)
    else:
        return derivative(f, x0, h, n - 1, args)
