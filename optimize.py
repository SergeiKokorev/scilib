from linalg import inverse, mult, add, norm


def polyder(x:float, p:list, n:int) -> float:
    '''
        Computes polynomial derivative of n-order at the specific point x
        ---------------------------------------------------
        Parameters:      x : float
                            The point to compute a polynomial
                        args : array_like
                            Cpolynomial coefficients a[0] * x**n + a[1] * x**(n - 1) + ... + a[n] = 0
                        n : int
                            Order of the derivative
        Returns:        result : float
                            The derivative of the specified order of a polynomial
    '''
    m = len(p) - 1        
    if n == 1:
        return sum([ai * (m - i) * x ** (m - i - 1) for i, ai in enumerate(p[:-1])])
    elif n == 0:
        return p(x, p)
    else:
        return polyder(x, [ai * (m - i) for i, ai in enumerate(p[:-1])], n - 1)


def synthetic(p:list, d:list) -> list[float]:
    '''
        Compute synthetic division of a polynomial with the specific coefficients p and denominator (x - d)
        --------------------------------------------------------------------------------------
        Parameters:     
        -----------
                        p : array_like
                            Coefficients of a polynomial a[0] * x**n + a[1] * x**(n - 1) + ... + a[n] = 0
                        d : float
                            A coefficient of linear denominator (x - d)
        Returns:
        --------
                        result : array_like
                            Coefficients of a polynomial after division of polynomial with size len(p) - 1
    '''
    res = []
    for i, ai in enumerate(p):
        if i == 0:
            res.append(ai)
        else:
            res.append(p[i] + d * res[i - 1])
    return res[:-1]


def newton_raphson(f:callable, x0:float=None, args:tuple=(), dx:float=1e-6, tol:float=1e-6, max_iter:int=800) -> float:

    '''
        Modified Newton-Raphson method for finding root of the function of one variable
        -------------------------------------------------------------------------------
        Parameters:
        -----------     
                        f : callable
                            The function root whose need to be found. The function is called
                            as f(x, *args)
                        x0 : float, optional
                            An initial guess of a root. If None it will be set to 0.0
                        dx : float, optional
                            Argument increment to find f'(x) and f''(x).
                            Default value 1e-6
                        tol : float, optional
                            Absolute error (xi+1 - xi). Stopping criterion.
                            Defaule value 1e-6
                        max_iter : int, optional
                            Maximum number of iterations.
                            Default value 800
        Returns:
        --------
                        result : float
                            The root of the function f
    '''

    def fprime(f, x, dx, args):
        '''
            Returns f'(x)
        '''
        return (f(x + dx, *args) - f(x - dx, *args)) / (2 * dx)

    def f2prime(f, x, dx, args):
        '''
            Returns f''(x)
        '''
        return (f(x + dx, *args) -2 * f(x, *args) + f(x - dx, *args)) / (dx ** 2)
    
    def uprime(f, x, dx, args):
        '''
            Returns u(x) / u'(x)
            Where u(x) = f(x) / f'(x)
        '''
        fp = fprime(f, x, dx, args)
        fx = f(x, *args)
        return fx * fp / (fp ** 2 - fx * f2prime(f, x, dx, args))

    if not x0: x0
    if max_iter == 0:
        raise RuntimeError('Maximum iteration exceeded. No solution found')    
    
    xn = x0 - uprime(f, x0, dx, args)
    es = xn - x0
    if abs(es) <= tol:
        return xn
    else:
        return newton_raphson(f, xn, args, dx, tol, max_iter-1)


def fsolve(func:callable, x0:list, args=(), dx:float=None, jac:callable=None, rtol:float=1e-6, max_iter:int=100, w:float=None) -> list[float]:
    
    '''
        Find roots of a function.
        Returns the roots of the (non-linear) equations defined by func(x) = 0 given a statring
        estimate
        ---------------------------------------------------------------------------------------
        Parameters:     
        -----------
                        func : callable f(x, *args)
                            A function that takes at least one argument, and returns a value 
                            of the same length
                        x0 : array_like, optional
                            The starting estimate for the roots of func(x) = 0.
                            If None will be set to 1.0 for all x
                        args : tuple, optional
                            Any extra arguments to func
                        dx : array_like, optional
                            Step size array. If None will be set to 1e-6 for each x
                        jac : callable, optional
                            Function to compute Jacobian matrix. If None will be estimated
                        rtol : float, optional
                            The calculation will terminate if the relative error between two
                            consecutive iterates is at most rtol. Default 1e-6
                        max_iter : integer, optional
                            Maximum number of iterations. Default 800
                        w : array_like, optional
                            Damping factor. If None estimated solution will not be damped
        Returns:        
        --------
                        x : array_like
                            The solution (or the result of last iteration for an unsuccessful call)
    '''


    n = len(x0)

    def jacobian_matrix(func, x, dx, args=()):
        jac = [[0.0 for j in range(n)] for i in range(n)]
        
        for i in range(n):
            for j in range(n):
                x1 = [x[k] if k != j else x[k] + dx[k] for k in range(n)]
                x2 = [x[k] if k != j else x[k] - dx[k] for k in range(n)]
                jac[i][j] = (func(x1, *args)[i] - func(x2, *args)[i]) / dx[j]
        return jac

    if not x0: x0 = [1.0 for i in range(n)]
    if not dx: dx = [1e-6 for i in range(n)]

    for i in range(max_iter):

        if not jac:
            jacobian = jacobian_matrix(func, x0, dx, args)
        else:
            jacobian = jac(x0, *args)
        
        fs = func(x0, *args)
        jacobian = inverse(jacobian)
        
        if not w:
            delx = mult(mult(-1, jacobian), fs)
        else:
            delx = mult(mult(mult(-1, jacobian), fs), w)
        
        xn = add(x0, delx)
        es = norm([(f0 - fn) ** 2 for f0, fn in zip(fs, func(xn, *args))]) / norm(fs)

        if abs(es) <= rtol:
            return xn
        else:
            x0 = xn

    print('Maximum iterration exceded. No solutions found')
    return xn


def root(p: list, tol:float=1e-6, max_iter:int=800) -> list[float]:

    '''
        Computes roots of a polynomial with coefficients p by using Laguerre's ,ethod.
        --------------------------------------------------
        Parameters:
        -----------
                        p : array_like
                            Coefficients of a polynomial a[0] * x**n + a[1] * x**(n - 1) + ... + a[n] = 0
                        tol : float, optional
                            Root searching stopping creteria. if P(x) <= tol, root is considered found.
                            Default 1e-6
                        max_iter : int
                            Maximum number of interations to find one of the equation roots.
                            Default 800
        Returns:        
        --------
                        result : array_like
                            Roots of equation P(x) = a[0] * x**n + a[1] * x**(n - 1) + ... + a[n] = 0
                            Size of the array equals degree of polynomial
    '''

    def poly(x, p):
        '''
            Computes polynomial at the point x
            ----------------------------------
            Parameters:     
            -----------
                            x : float
                                Point to compute polynomial
                            p : array_like
                                Polynomial coefficients  a[0] * x**n + a[1] * x**(n - 1) + ... + a[n] = 0
            Returns:        
            --------
                            result : float
                                The value of the polynomial at the point x
        '''
        n = len(p) - 1
        return sum([ai * x ** (n - i) for i, ai in enumerate(p)])

    n = len(p) - 1
    roots = [0.0 for i in range(n)]
    p = [ai for ai in p]
    x0 = -1.0

    for i in range(n):
        for k in range(max_iter):

            if k == max_iter - 1:
                print('Maximum number of iterations exceded. No root found.')
                break
            
            if abs(poly(x0, p)) <= tol:
                roots[i] = x0
                p = synthetic(p, x0)
                break
            
            G = polyder(x0, p, 1) / poly(x0, p)
            H = G ** 2 - polyder(x0, p, 2) / poly(x0, p)
            denom = [G + ((n - 1) * (n * H - G ** 2)) ** 0.5, G - ((n - 1) * (n * H - G ** 2)) ** 0.5]

            if (denom[0].real ** 2 + denom[0].imag ** 2) ** 0.5 > (denom[1].real ** 2 + denom[1].imag ** 2) ** 0.5:
                denom = denom[0]
            else:
                denom = denom[1]

            a = n / denom
            x0 -= a

    return roots
