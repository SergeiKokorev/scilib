from math import floor
from copy import deepcopy
from utils import (
    is_square, swap_rows, 
    separate, compare, join,
    diag
)


def zeros(size: tuple | int):
    if hasattr(size, '__iter__'):
        return [[0.0 for i in range(size[0])] for j in range(size[1])]
    elif isinstance(size, int):
        return [0.0 for i in range(size)]
    else:
        raise TypeError('Wrong size type. Size must be integer or tuple of integers')



def identity(n: int) -> list:
    return [[float(i == j) for i in range(n)] for j in range(n)]


def norm(m: list) -> float:
    '''
        Compute Euclidian norm of vector or matrix
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


def distance(v1: list, v2: list) -> float:
    '''
        Returns distance between two vectors v1 and v2
    '''
    if not isinstance(v1, list) or not isinstance(v2, list):
        raise TypeError('Vectors must be list type')    
    
    if len(v1) != len(v2):
        raise ValueError('Vectors must be same dimention')
    
    return sum([(v1i - v2i) ** 2 for v1i, v2i in zip(v1, v2)]) ** 0.5


def transpose(m: list, copy: bool = False) -> list:

    '''
        Transpose matrix or vector
        Parameters  copy: bool (optional, default False)
                        make copy list m or change list m
    '''

    res = m if not copy else deepcopy(m)

    if not hasattr(res, '__iter__'):
        raise TypeError('Argument has to be iterable')
    
    if hasattr(res[0], '__iter__'):
        tmp = deepcopy(res)
        for row in range(len(res)):
            for col in range(len(res[0])):
                res[row][col] = tmp[col][row]
    else:
        tmp = deepcopy(res)
        res = []
        for row in range(len(tmp)):
            res.append([tmp[row]])
    return res


def dot(m1: list, m2: list) -> float | bool:

    '''
        Returns dot product of two matrices or vectors
    '''
    if not hasattr(m1) or not hasattr(m2):
        raise TypeError('Arguments has to iterable')

    if len(m1) != len(m2):
        raise ValueError('Shapes of matrices not aligned')
    
    if isinstance(m1[0], float | int) and isinstance(m1[0], float | int):
        return sum([m1i * m2i for m1i, m2i in zip(m1, m2)])
    elif hasattr(m1[0], '__iter__') and hasattr(m2[0], '__iter__'):
        res = 0.0
        for m1i, m2i in zip(m1, m2):
            res += sum(m1ij * m2ij for (m1ij, m2ij) in zip(m1i, m2i))
        return res


def mult(m1: list | float, m2: list | float) -> list:

    '''
        Return matrix multiplication
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

    # Multiply a matrix by a mtrix
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


def pow(m: list, n: int) -> list:

    '''
        Returns matrix m to the power n
    '''
    if n == 1:
        return m
    else:
        p = pow(m, floor(n / 2))
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
        if isinstance(m1[0], int | float) and isinstance(m2[0], int | float):
            return [m1i + m2i for m1i, m2i in zip(m1, m2)]
        elif hasattr(m1[0], '__iter__') and hasattr(m2[0], '__iter__'):
            return [add(m1[i], m2[i]) for i in range(len(m1))]   


def sub(m1: list, m2: list) -> list:
    '''
        Matrix substraction
    '''
    return add(m1, mult((-1), m2))


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
        tinv = add(pow(mult(mult((-1), lam), tu), i), tinv)
    
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


def det(m: list, mul: int = 1.0) -> float | bool:

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


def lu(a: list, permute_l: bool=False) -> tuple:

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
    P = identity(n) # pivot matrix
    U = deepcopy(a) # upper triangular matrix
    L = identity(n) # lower triangular matrix

    for k in range(n):
        j = max(range(k, n), key=lambda i: abs(U[i][k]))
        I = identity(n)
        I[k], I[j] = I[j], I[k]
        P = mult(P, I)
        U[k], U[j] = U[j], U[k]
        
        for r in range(k + 1, n):
            if U[r][k] != 0.0:
                ratio = (-1) * U[k][k] / U[r][k]
                U[r] = [ui + uk / ratio for ui, uk in zip(U[r], U[k])]

    if permute_l:
        L = mult(a, inverse_tri(U))
        return mult(P, L), U
    else:
        L = mult(mult(transpose(P, copy=True), a), inverse_tri(U))
        return P, L, U


def solve(a: list, b: list) -> list:
    '''
        Solves the linear equation a x = b for the unknows x for square a matrix
        ------------------------------------------------------------------------
        Parameters:     a(N, N) array_like
                            Square input matrix
                        b: (N) array_like
                            Input data for right hand side
        
        Returns:        x: (N) assray_like
                            The solution array

        Raises:         ValueErros
                            If size mismatches detected or input a is not square
    '''

    if not is_square(a) : raise ValueError('Matrix a is not square.')
    if (n:=len(a)) != (m:=len(b)) : raise ValueError('Matrix a is mismatched vector b.')

    y = zeros(n)
    x = zeros(n)

    P, L, U = lu(a)
    bp = mult(b, inverse(P))

    # Forward substitution [L]{y} = {b}, {y} = [U]{x}
    for i in range(n):
        y[i] = (1 / L[i][i]) * (bp[i] - sum(L[i][j] * y[j] for j in range(i)))
    
    # Backward substitution [L]{x} = {y}
    for i in range(n-1, -1, -1):
        x[i] = (1 / U[i][i]) * (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n)))
    
    return x
