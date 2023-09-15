from copy import deepcopy

import linalg



def swap_rows(m: list, i: int, j: int, copy: bool = False) -> list:
    '''
        Swap i and j rows of matrix m
        -----------------------------
        Parameters:     m: list
                            matrix to be changed
                        i, j : int
                            rows to be swapped
                        copy: bool (optional)
                            if whether to create a deepcopy of matrix m.
                            Default initial matrix will be changed
    '''
    res = m if not copy else deepcopy(m)
    if i == j : return res
    tmp = [mi for mi in res[j]]
    res[j] = [mi for mi in res[i]]
    res[i] = [mj for mj in tmp]
    return res


def swap_cols(m: list, i: int, j: int, copy: bool = False) -> list:
    '''
        Swap i and j rows of matrix m
        -----------------------------
        Parameters:     m: list
                            matrix to be changed
                        i, j : int
                            cols to be swapped
                        copy: bool (optional)
                            if whether to create a deepcopy of matrix m.
                            Default initial matrix will be changed
    '''
    res = m if not copy else deepcopy(m)
    if i ==j : return res
    tmp = []
    rows = len(res)
    cols = len(res[0])
    for row in range(rows):
        tmp.append(res[row][j])
    for row in range(rows):
        res[row][j] = res[row][i]
    for row in range(rows):
        res[row][i] = tmp[row]
    return res


def max_element(m: list, row: int, col: int) -> tuple:
    '''
        Find max value and return index and maximal value
        -------------------------------------------------
        Parameters:     m: list
                            Matrix to find maximal element
                        row, col: int
                            Initial row and col to find element and index
    '''
    n = len(m)
    k = len(m[0])
    idx = (row, col)
    max_arg = abs(m[row][col])
    for r in range(row, n):
        for c in range(col, k):
            if (arg := abs(m[r][c])) >= max_arg:
                max_arg = arg
                idx = (r, c)
    return idx, max_arg


def separate(m: list, col: int) -> list:
    
    nrows = len(m)
    cols = len(m[0])
    ncols1 = col
    ncols2 = cols - col

    r1 = linalg.zeros(size=(ncols1, nrows))
    r2 = linalg.zeros(size=(ncols2, nrows))

    for r in range(nrows):
        for c in range(cols):
            if c < col:
                r1[r][c] = m[r][c]
            else:
                r2[r][c - col] = m[r][c]
    
    return [r1, r2]


def is_square(m: list) -> bool:
    if len(m) == 1 and isinstance(m[0], (int, float)):
        return True
    if not (len(m) == len(m[0])):
        return False
    else:
        return True
    

def is_diagonal_dominant(m: list) -> bool:

    n = len(m)
    r = []
    for i in range(n):
        r.append(abs(m[i][i]) >= sum([m[i][j] for j in range(n) if j != i]))
    return all(r)


def join(m1: list, m2: list) -> list:

    out = []
    for row1, row2 in zip(m1, m2):
        tmp = [r for r in row1]
        for val in row2:
            tmp.append(val)
        out.append(tmp)

    return out


def compare(m1: list, m2: list) -> bool:

    if not (len(m1) - len(m2)) or not (len(m1[0]) - len(m2[0])):
        return False
        
    for m1_row, m2_row in zip(m1, m2):
        for el_1, el_2 in zip(m1_row, m2_row):
            if (el_1 - el_2) > 10 ** -6:
                return False

    return True
