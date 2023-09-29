def solve_ipv(fun, tspan, y0, method='RK45', teval=None, args=()):
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
                        teval: array_like or None, optional
                            Times at which to store the computed solution, must be lie
                            within tspan. If None (default), use points selected by solver
        Rerturns:       t: ndarray, shape(n_points)
                        y: ndarray, shape(n, n_points)
                            
    '''

    yf = y0
    n = len(y0)
    tf = tspan[0]
    tsol = [tf]
    ysol = [[y0i] for y0i in yf]
    teval = 100 if not teval else teval
    h = (tspan[1] - tspan[0]) / teval
    
    for s in range(teval):
        k1 = fun(tf, yf, *args)
        k2 = fun(tf + h / 2, [yfi + h * k1i / 2 for yfi, k1i in zip(yf, k1)], *args)
        k3 = fun(tf + h / 2, [yfi + h * k2i / 2 for yfi, k2i in zip(yf, k2)], *args)
        k4 = fun(tf + h, [yfi + h * k3i for yfi, k3i in zip(yf, k3)], *args)
        yf = [
            yfi + (h / 6) * (k1i + 2 * k2i + 2 * k3i + k4i) 
            for yfi, k1i, k2i, k3i, k4i in zip(yf, k1, k2, k3, k4)
        ]
        tf += h
        tsol.append(tf)
        for i in range(n):
            ysol[i].append(yf[i])
    
    return tsol, ysol
