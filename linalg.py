from math import floor, hypot
from copy import deepcopy
from utils import (
    is_square, swap_rows, 
    separate, compare, join,
    diag
)


def vstack(m: list, v: list):
    if len(m) == 0:
        for vi in v:
            m.append([vi])
    else:
        for i, vi in enumerate(v):
            m[i].append(vi)


def zeros(size: tuple | int):
    '''
        size : tuple(row, col)
    '''
    if hasattr(size, '__iter__'):
        return [[0.0 for i in range(size[1])] for j in range(size[0])]
    elif isinstance(size, int):
        return [0.0 for i in range(size)]
    else:
        raise TypeError('Wrong size type. Size must be integer or tuple of integers')



def identity(n: int) -> list:
    return [[float(i == j) for i in range(n)] for j in range(n)]


def norm(m: list) -> float:
    '''
        Compute Euclidian norm of a vector or a matrix
    '''

    if not hasattr(m, '__iter__'):
        raise TypeError('Argument has to be iterable')

    if all([isinstance(mi, (float, int)) for mi in m]):
        return sum([mi ** 2 for mi in m]) ** 0.5
    
    if all([hasattr(mi, '__iter__') for mi in m]):
        s = 0.0
        for row in m:
            s += sum([col ** 2 for col in row])
        return s ** 0.5


def normalize(v: list) -> list:

    n = norm(v)
    return [vi / n for vi in v]


def distance(v1: list, v2: list) -> float:
    '''
        Returns distance between two vectors v1 and v2
    '''
    if not hasattr(v1, '__iter__') or not hasattr(v2, '__iter__'):
        raise TypeError('Vectors must be list type')    
    
    if len(v1) != len(v2):
        raise ValueError('Vectors must be same dimension')
    
    return sum([(v1i - v2i) ** 2 for v1i, v2i in zip(v1, v2)]) ** 0.5


def conj(a):
    
    if any([isinstance(ai, complex) for ai in a]):
        try:
            a = [complex(ai.real, ai.imag) for ai in a]
        except ValueError as er:
            raise(f'Unsupported type for conjugation.')
    elif all([isinstance(ai, (float, int)) for ai in a]):
        return a

    if isinstance(a, complex):
        return complex(a.real, (-1) * a.imag)
    elif hasattr(a, '__iter__') and all([isinstance(ai, complex) for ai in a]):
        return [complex(ai.real, (-1) * ai.imag) for ai in a]
    elif hasattr(a, '__iter__') and all([isinstance(ai, (float, int)) for ai in a]):
        return a
    elif hasattr(a, '__iter__') and hasattr(a[0], '__iter__'):
        return [conj(ai) for ai in a]
    else:
        TypeError(f'Unsupported type for conjugation. Expected (float, int, complex). Given {type(a)}')


def transpose(m: list, copy: bool = False) -> list:

    '''
        Transpose matrix or vector
        Parameters:     m: matrix to transpose
                        copy: bool (optional, default False)
                            make copy list m or change list m
        Returns:        result: list
                            Transposed matrix
    '''

    res = m if not copy else deepcopy(m)

    if not hasattr(res, '__iter__'):
        raise TypeError('Argument has to be iterable')
    
    if hasattr(res[0], '__iter__'):
        return list(map(list, zip(*m)))
    else:
        tmp = deepcopy(res)
        res = []
        for row in range(len(tmp)):
            res.append([tmp[row]])
    return res


def dot(m1, m2) -> float | bool:

    '''
        Returns dot product of two matrices or vectors
        ----------------------------------------------
        Parameters      m1 : array_like | float
                            array_like size of (n x m) or float
                        m2 : array_like | float
                            array_like size of (m x k) of float
        Returns         res : array_like | float
    '''
    if isinstance(m1, (float, int)) and isinstance(m2, (float, int)):
        return m1 * m2
    elif isinstance(m1, (float, int)) and hasattr(m2, '__iter__'):
        if all([isinstance(m2i, (float, int)) for m2i in m2]):
            return [m1 * m2i for m2i in m2]
        elif all([hasattr(m2i, '__iter__') for m2i in m2]):
            return [dot(m1, m2i) for m2i in m2]
    elif hasattr(m1, '__iter__') and isinstance(m2, (float, int)):
        if all([isinstance(m1i, (float, int)) for m1i in m1]):
            return [m1i * m2 for m1i in m1]
        elif all([hasattr(m1i, '__iter__') for m1i in m1]):
            return [dot(m1i, m2) for m1i in m1]
    elif hasattr(m1, '__iter__') and hasattr(m2, '__iter__'):
        if all([isinstance(m1i, (float, int)) for m1i in m1]):
            if all([isinstance(m2i, (float, int)) for m2i in m2]):
                return sum([m1i * m2i for m1i, m2i in zip(m1, m2)])
            elif all([hasattr(m2i, '__iter__') for m2i in m2]):
                return [dot(m1, m2i) for m2i in transpose(m2, copy=True)]
        elif all([hasattr(m1i, '__iter__') for m1i in m1]):
            if all([hasattr(m2i, '__iter__') for m2i in m2]):
                return [dot(m1i, m2) for m1i in m1]
            elif all([isinstance(m2i, (float, int)) for m2i in m2]):
                return [dot(m1i, m2) for m1i in m1]
    else:
        raise RuntimeError('Unsupported type.')



def mult(m1: list | float, m2: list | float) -> list:

    '''
        Return multiplication of matrices, matrices and scalars
    '''

    # Multiplication scalar by scalar
    if isinstance(m1, float | int) and isinstance(m2, float | int):
        return m1 * m2

    # Multiply a matrix by a scalar
    if isinstance(m1, int | float) and hasattr(m2, '__iter__'):
        if isinstance(m2[0], int | float): 
            return [m1 * m2i for m2i in m2]
        else:
            return [mult(m1, m2[i]) for i in range(len(m2))]

    if isinstance(m2, int | float) and hasattr(m1, '__iter__'):
        if isinstance(m1[0], int | float): 
            return [m2 * m1i for m1i in m1]
        else:
            return [mult(m2, m1[i]) for i in range(len(m1))]

    # Multiply a matrix by a vector
    if isinstance(m1[0], (float, int)) and hasattr(m2[0], '__iter__'):
        res = [0.0 for i in range(len(m2))]
        for i, el in enumerate(m2):
            res[i] = sum([eli * m1i for eli, m1i in zip(el, m1)])
        return res
        
    if isinstance(m2[0], (float, int)) and hasattr(m1[0], '__iter__'):
        res = [0.0 for i in range(len(m1))]
        for i, el in enumerate(m1):
            res[i] = sum([eli * m2i for eli, m2i in zip(el, m2)])
        return res

    # Multiply a matrix by a matrix
    n, m, p, q = len(m1), len(m1[0]), len(m2), len(m2[0])

    if m != p:
        raise ValueError('Number of rows of first matrix mus be equal number of columns second matrix')
    
    res = [[0.0 for i in range(q)] for j in range(n)]
    for i in range(n):
        for j in range(q):
            summ = 0.0
            for k in range(p):
                summ += m1[i][k]  * m2[k][j]
            res[i][j] = summ

    return res


def pow_mat(m: list, n: int) -> list:

    '''
        Returns matrix m to the power n
    '''

    if n == 1:
        return m
    else:
        p = pow_mat(m, floor(n / 2))
        if n % 2 == 0:
            return mult(p, p)
        else:
            return mult(p, mult(p, m))


def add(m1: list, m2: list) -> list:
    '''
        Matrix addition
    '''
    if isinstance(m1, int | float) and isinstance(m2, int | float):
        return m1 + m2

    if isinstance(m1, int | float) and hasattr(m2, '__iter__'):
        if isinstance(m2[0], int | float): 
            return [m1 + m2i for m2i in m2]
        else:
            return [add(m1, m2[i]) for i in range(len(m2))]

    if isinstance(m2, int | float) and hasattr(m1, '__iter__'):
        if isinstance(m1[0], int | float): 
            return [m2 + m1i for m1i in m1]
        else:
            return [add(m2, m1[i]) for i in range(len(m1))]
    
    if hasattr(m1, '__iter__') and hasattr(m2, '__iter__'):
        if (all([isinstance(m1i, int | float) for m1i in m1])
            and all([isinstance(m2i, int | float) for m2i in m2])):
            return [m1i + m2i for m1i, m2i in zip(m1, m2)]
        elif (all([hasattr(m1i, '__iter__') for m1i in m1])
              and all([hasattr(m2i, '__iter__') for m2i in m2])):
            return [add(m1[i], m2[i]) for i in range(len(m1))]   


def sub(m1: list, m2: list) -> list:
    '''
        Matrix substraction
    '''
    return add(m1, dot((-1), m2))


def upper(m: list, copy: bool = False):

    rows = len(m)
    c = m if not copy else deepcopy(m)
    cols = len(c[0])
    
    for k in range(rows):
        pivot = c[k][k]
        for i in range(k+1, rows):
            pivot_col = c[i][k]
            for j in range(cols):
                pivot_row = c[k][j]
                c[i][j] = c[i][j] - pivot_col * pivot_row / c[k][k]

    return c


def inverse_tri(t: list) -> list:
    '''
        Invertes triangular matrix    
    '''
    if not is_square(t): return False

    n = len(t)
    lam = diag(t)
    tu = sub(t, lam)
    for i in range(n):
        lam[i][i] = 1 / lam[i][i]
    
    tinv = identity(n)
    for i in range(1, n):
        tinv = add(pow_mat(mult(mult((-1), lam), tu), i), tinv)
    
    return mult(tinv, lam)


def inverse(m: list, diag: bool = False) -> list | bool:
    '''
        Inverse matrix m
            Parameters:     m: list
                                Matrix to inverse. Must be square matrix
                            diag: bool, optional
                                Is the matrix m diagonal matrix. Default False
            Returns:        result: list | bool
                                Return inverse matrix or if inversion not succeed
                                False
    '''

    def find_row_with_max_element(m, col_num: int=0, starting_row: int=0):
        tmp = m[starting_row][col_num]
        row_idx = starting_row
        for k in range(starting_row+1, len(m)):
            if abs(m[k][col_num]) > abs(tmp):
                row_idx = k
                tmp = m[k][col_num]
        return row_idx

    if not is_square(m):
        return False

    identity_matrix = identity(len(m))
    num_rows = len(m) - 1
    joint_matrix = join(m, identity_matrix)

    flag = False
    count = 0
    max_count = 100

    if diag:
        inverse_matrix = deepcopy(m)
        if all([inverse_matrix[i][i] != 0 for i in range(len(inverse_matrix))]):
            for i in range(len(inverse_matrix)):
                inverse_matrix[i][i] = 1 / inverse_matrix[i][i]
            return inverse_matrix
        else:
            raise ZeroDivisionError('All main diagonal elements must be non-zero.')

    while not flag and count < max_count:
        for i in range(num_rows + 1):
            if joint_matrix[i][i] == 0.0:
                max_el_idx = find_row_with_max_element(joint_matrix, i, i)
                joint_matrix = swap_rows(joint_matrix, i, max_el_idx)
            div_e = joint_matrix[i][i]
            factor = 1 / div_e
            joint_matrix[i] = [e * factor for e in joint_matrix[i]]
            for row in range(0, num_rows+1):
                if row != i:
                    if joint_matrix[row][i] != 0:
                        sub = (-1) * joint_matrix[row][i]
                        row_add = [el * sub for el in joint_matrix[i]]
                        joint_matrix[row] = [e1 + e2 for e1, e2 in zip(row_add, joint_matrix[row])]
    
        identity_matrix, inverse_matrix = separate(m=joint_matrix, col=num_rows+1)
        if not compare(identity(num_rows+1), identity_matrix):
            flag = True
        count += 1
    
    if not flag:
        return False
    else:
        return inverse_matrix


def det(m: list, mul: int = 1.0, lu: bool = True) -> float | bool:

    '''
        Returns determinant of the matrix m
        ----------------------------------
            Parameters:     m: list
                                Matrix which determinant need to be found
                            mul: int
                                Temporary variable for recursion
    '''

    width = len(m)

    if width == 1:
        return mul * m[0][0]
    else:
        sign = -1
        answer = 0
        for col in range(width):
            mi = []
            for row in range(1, width):
                buff = []
                for k in range(width):
                    if k != col:
                        buff.append(m[row][k])
                mi.append(buff)
            sign *= -1
            answer = answer + mul * det(mi, sign * m[0][col])
    return answer


def backward_sub(a, b):

    n = len(b)
    x = [0.0 for i in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] / a[i][i]) - (1 / a[i][i]) * sum([a[i][j] * x[j] for j in range(i+1, n)])
    
    return x


def gaussian(A, b, copy=True):

    '''
        Computes Gaussian elimination.
        --------------------------------
        Parameters:     A: array_like:
                            Square matrix of coefficients size (N, N)
                        b: array_like:
                            RHS of the system of equations, size(N,)
                        copy: bool, optional
                            Make copy of matrix A and vector b
        Returns:        result: tuple
                            A, b
                            A is a upper triangular matrix
                            b is new vector of coefficients
        Raise error:    TypeError:
                            If the matrix A is not square
                        RuntimeError:
                            If the len(A) != len(b)
                        TypeError:
                            If A contains row with all elements equal zero
    '''

    a = deepcopy(A) if copy else A
    b = deepcopy(b) if copy else b

    if not is_square(a):
        raise TypeError('During Gaussian elimination an error occurred. Matrix A must be square.')
    
    if len(a) != len(b):
        raise RuntimeError('During Gaussian elimination an error occurred. Number of rows of matrix A' \
                           ' must be equal number of rows vector b')

    if any([all([row[i] == 0 for i in range(len(a))]) for row in a]):
        raise TypeError('During Gaussian elimination an error occurred. Matrix A contains zero row.')
    
    n = len(a)

    for j in range(len(a)):

        k = max(range(j, n), key=lambda i: abs(a[i][j]))

        a[j], a[k] = a[k], a[j]
        b[j], b[k] = b[k], b[j]

        if not a[j][j]:
            continue
        else:
            pivot = a[j][j]

        for i in range(j + 1, n):
            ratio = a[i][j] / pivot
            b[i] -= b[j] * ratio
            for h in range(j, n):
                a[i][h] -= a[j][h] * ratio

    for i in range(n):
        pivot = a[i][i]
        b[i] /= pivot
        a[i] = [ai / pivot for ai in a[i]]

    return a, b


def lu(a: list) -> tuple:

    '''
        Computes LU decomposition of a matrix with partial pivoting
        ----------------------------------------------------------
        Parameters:     m: (M, N) array_like
                            Array to decompose
                        permute_l: bool, optional
                            Perfome the multiplication P*L (Default do not perfome)
        Returns:        (if permute_l is False)
                        p: array_like
                            Permutation array or vectors depending on p_indices
                        l: array_like
                            Lower triangulat matrix
                        u: array_like
                            Upper triangular matrix
                        (if permut_l is True)
                        pl: array_like
                            Permute L matrix
                        u: array_like
                            Upper diagonal matrix
    '''

    n = len(a)
    L = identity(n)
    P = identity(n)
    U = deepcopy(a)

    for i in range(n):

        id_max = max(range(i, n), key=lambda k: abs(U[k][i]))
        
        I = identity(n)
        I[i], I[id_max] = I[id_max], I[i]
        U[id_max], U[i] = U[i], U[id_max]
        P = mult(P, I)
        L = mult(I, mult(L, I))

        pivot = U[i][i]
        for j in range(i + 1, n):
            c = U[j][i] / pivot

            for k in range(i, n):
                U[j][k] = U[j][k] - c * U[i][k]
                L[j][k] = L[j][k] + c * L[i][k]

    return P, L, U


def proj(v, u):
    '''
        Returns projection v on u
    '''
    return dot(dot(v, u) / dot(u, u), u)


def gsa(s):

    '''
        Gram-Schmidt algorithm
        ----------------------
        Parameter   s : array_like
                        Set of vectors
        Returns     u : array_like
                        Orthonormal basis
    '''

    m = len(s)
    n = len(s[0])
    u = zeros((m, n))
    st = transpose(s, copy=True)
    u[0] = [si for si in st[0]]

    for i in range(1, n):
        u[i] = sub(st[i], proj(st[i], u[0]))
        for j in range(1, i):
            u[i] = sub(u[i], proj(u[i], u[j]))
    return transpose(u)


def arnoldi(A, b, n=1):

    '''
        Compute a basis of the (n + 1)-Krylov subspace of the matrix A.
        ---------------------------------------------------------------
        Parameters
        ----------
        A   :   array_like
                An m x m array
        b   :   array_like
                Initial vector (length m)
        n   :   int
                One less than the dimension of the Krylov subspace, or equivalently
                the *degree* of the Krylov space. Must be >= 1
        Returns
        -------
        Q   :   array_like
                An m x (n + 1) array, where the columns are an orthonormal basis of
                the the Krylov subspace
        h   :   array_like
                An (n + 1) x n array. A on basis Q. It is upper Hessenberg.
    '''

    eps = 1e-12
    m = len(A)
    a = deepcopy(A)

    q = zeros((n+1, m))
    h = zeros((n, n+1))

    q[0] = normalize([bi for bi in b])
    
    for k in range(1, n + 1):
        q[k] = dot(a, q[k-1])
        for j in range(k):
            h[k-1][j] = dot(conj(q[j]), q[k])
            q[k] = sub(q[k], dot(h[k-1][j], q[j]))
        h[k-1][k] = norm(q[k])
        
        if h[k-1][k] > eps:
            q[k] = normalize(q[k])
        else:
            return transpose(q[:k]), transpose(h[:k])[:k]
    
    return transpose(q), transpose(h)


def qr(m: list) -> tuple:

    '''
        Computes QR decomposition of a square matrix a by usin Givens rotation
        ----------------------------------------------------------------------
        Parameters:     m: (N, N) array_like
                            Array to decompose
        Returns:        tuple of Q, R
                            where Q is an orthogonal matrix
                                  R is an upper triangular matrix
                                  A = QR
    '''

    n = len(m)
    R = deepcopy(m)
    Q = zeros(size=(n, n))

    for j in range(n):
        for i in range(j + 1, n):

            g = zeros(size=(n, n))

            a, b = R[j][j], R[i][j]
            r = hypot(a, b)
            c = a / r
            s = -b / r
            g[i][j] = s
            g[j][i] = -s
            for k in range(n):
                g[k][k] = c if (k == j or k == i) else 1

            Q = transpose(g, copy=True) if i == 1 else mult(Q, transpose(g, copy=True))
            R = mult(g, R)
    
    return Q, R


def eig(a):

    '''
    Compute the eigenvalues and eigenvectors of a square matrix array

    Parameters
    ----------
    a : array_like size of m x m
        Matrices for which the eigenvalues and eigenvectors will be
        computed
    
    Returns
    -------
    eigval : array_like size of m
            The eigenvalues
    eigvec : array_like size of m x m
            The normalized eigenvectors, such that the column ``eigenvectors[:][i]``
            is the eigenvector corresponding to the eigenvalue ``eigenvalue[i]``

    Raises
    ------
    RuntimeError
        If the eigenvalue computation does not converge
    '''

    m = deepcopy(a)
    for i in range(800):
        if all([abs(m[i][j]) < 1e-10 for i in range(1, len(m)) for j in range(i)]):

            eigval = [m[i][i] for i in range(len(m))]
            I = identity(len(m))
            eigvec = []
            rhs = [0.0 for i in range(len(m) - 1)]
            rhs.append(1)
            for lam in eigval:
                lhs = sub(a, dot(lam, I))
                eigvec.append(normalize(solve(lhs, rhs)))
            return (eigval, transpose(eigvec))
        else:
            q, r = qr(m)
            m = mult(mult(transpose(q, copy=True), m), q)

    raise RuntimeError('Eigen runtime error. Maximum numbers of iterations exceeded. No solution found.')


def solve(a: list, b: list, tridiagonal=False) -> list:
    '''
        Solves the linear equation a x = b for the unknows x for square a matrix
        ------------------------------------------------------------------------
        Parameters:     a(N, N) array_like
                            Square input matrix
                        b: (N) array_like
                            Input data for right hand side
                        tridiagonal: bool, optional
                            Wether the matrix a tridiagonal or not. Default False
        
        Returns:        x: (N) assray_like
                            The solution array

        Raises:         ValueErros
                            If size mismatches detected or input a is not square
    '''

    if not is_square(a) : raise ValueError('Matrix a is not square.')
    if (n:=len(a)) != (m:=len(b)) : raise ValueError('Matrix a is mismatched vector b.')

    if tridiagonal:
        bm = [a[i][i] for i in range(n) for j in range(n) if i == j]
        am = [a[i+1][i] for i in range(n - 1) for j in range(n) if i == j]
        cm = [a[i][i+1] for i in range(n) for j in range(n - 1) if i == j]
        dm = deepcopy(b)
        x = zeros(n)

        for i in range(n - 1):
            w = am[i] / bm[i]
            bm[i + 1] -= w * cm[i]
            dm[i + 1] -= w * dm[i]
        x[n - 1] = dm[n - 1] / bm[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = (dm[i] - cm[i] * x[i + 1]) / bm[i]
        
        return x

    y = zeros(n)
    x = zeros(n)

    P, L, U = lu(a)
    bp = mult(b, transpose(P, copy=True))

    # Forward substitution [L]{y} = {b}, {y} = [U]{x}
    for i in range(n):
        y[i] = (1 / L[i][i]) * (bp[i] - sum(L[i][j] * y[j] for j in range(i)))
    
    # Backward substitution [L]{x} = {y}
    for i in range(n-1, -1, -1):
        x[i] = (1 / U[i][i]) * (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n)))
    
    return x
