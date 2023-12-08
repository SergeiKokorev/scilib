from optimize import newton_raphson2


def rk45(y, fun, yn, t, h):
    
    s1 = yn + (h / 6) * (fun(t, yn) + fun(t + h, y))
    s2 = (2 * h / 3) * fun(t + h / 2, 0.5 * (yn + y) + (h / 8) * (fun(t, yn) - fun(t + h, y)))
    return s1 + s2


def solve_ipv(fun, tspan, y0, method='RK45', args=()):
    '''
        Solve an initial value problem for ODE
        --------------------------------------
        Parameters:     fun: callable
                            Right-hand side of the ODE: the time derivative
                            of the state y at time t. The calling signature
                            func(t, y), where t is a scalar and y is an 
                            ndarray with len(y) = len(y0). func must return
                            an array of the same shape as y.
                        tspan: 2-member sequence
                            Interval of integration (t0, tf)
                        y0: array_like, shape (n,)
                            Initial state.
                        intervals: integer, optional
                            Number of intervals. Default 100
                        method: string, optional
                            Integration method to use:
                                *   'RK45' (default): Explicit Runge-Kutta method of
                                    order 4
        Rerturns:       t: ndarray, shape(n_points)
                        y: ndarray, shape(n, n_points)
                            
    '''

    th = (tspan[1] - tspan[0]) / 100
    tsol = [tspan[0] + i * th for i in range(101)]

    n = len(y0)
    m = len(tsol)
    ysol = [[0.0 for i in range(m)] for j in range(n)]
    for i in range(n):
        ysol[i][0] = y0[i]

    for i in range(1, len(tsol)):
        h = tsol[i] - tsol[i - 1]
        tf = tsol[i - 1]
        yf = [yi[i - 1] for yi in ysol]
        k1 = fun(tf, yf, *args)
        k2 = fun(tf + h / 2, [yfi + h * k1i / 2 for yfi, k1i in zip(yf, k1)], *args)
        k3 = fun(tf + h / 2, [yfi + h * k2i / 2 for yfi, k2i in zip(yf, k2)], *args)
        k4 = fun(tf + h, [yfi + h * k3i for yfi, k3i in zip(yf, k3)], *args)
        yf = [
            yfi + (h / 6) * (k1i + 2 * k2i + 2 * k3i + k4i) 
            for yfi, k1i, k2i, k3i, k4i in zip(yf, k1, k2, k3, k4)
        ]

        for j in range(n):
            ysol[j][i] = yf[j]
    
    return tsol, ysol


def solve_bvp(fun, bc, x, y, max_it=10):

    poly = []

    if max_it <= 0:
        print('Maximum number of iterations exeeded. No solution found.')
        return None

    n = len(x)
    ysol = []

    for i in range(n - 1):
        h = x[i + 1] - x[i]
        ysol.append(newton_raphson2(rk45, y[:, i + 1], args=(fun, y[:, i], x[i], h)))
    


