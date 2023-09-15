from copy import deepcopy
from utils import (
    is_square, max_element, swap_rows, 
    separate, compare, join
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


def inverse(m: list, diag: bool = False) -> list | bool:
    '''
        Inverse matrix m
            Parameters:     m: list
                                Matrix to inverse. Must be square matrix
                            diag: bool
                                Is the matrix m diagonal matrix
            Returns:        result: list | bool
                                Return inverse matrix or if inversion not succeed
                                False
    '''

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
                max_el_idx = max_element(joint_matrix, i, i)[0][0]
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
        Return determinant of the matrix m
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
