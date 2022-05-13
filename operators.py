#!/usr/bin/env python
#===========================================================================
#
#          FILE:  operators.py
# 
#         USAGE:  import operators as op (from hib.py)
# 
#   DESCRIPTION:  math/logic operations for hibachi via deap 
# 
#  REQUIREMENTS:  a: numeric
#                 b: numeric
#                     not needed on unary operations
#        UPDATE:  Floats are dealt with as necessary for functions
#                 that require ints
#                 170216: added try/except to safediv()
#                 170626: added equal and not_equal operators
#        AUTHOR:  Peter R. Schmitt (@HOME), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.3
#       CREATED:  09/29/2016 10:39:21 EDT
#      REVISION:  Mon Jun 26 15:07:31 EDT 2017
#===========================================================================
import numpy as np
from hashlib import sha224
def hash(array): return(sha224(array.data.tobytes()).hexdigest())
import pandas as pd
from os.path import isdir
from os import mkdir
from time import time
from copy import deepcopy as COPY
#largest factorial less than numpy's 64 bit integer maximum 
FACTMAX = 20
MAX = np.product(FACTMAX  - np.arange(0, FACTMAX, dtype = np.float32))
import pdb

import IO
options = IO.get_arguments()
debug_out = options['debug_out']
debug_in = options['debug_in']
sigfigs = 6
i = 0
if debug_out != None:
    debug_vals = open(debug_out + ".txt", "w")
    debug_vals.write("hash1\thash2\tindex\n")
if debug_in != None:
    input_hash_log = pd.read_csv(debug_in + ".txt", delimiter = "\t", dtype = str)
###################### BASIC OPERATORS #################################

def round_figures(num, n):

    val = str(num)
    if '-' in val: n += 1
    if 'e' not in val:
        if '.' in val:
            left, right = val.split('.')
            val2 = str(len(left) - 1)
            if '-' in left: val2 = str(int(val2) - 1)
            val1 = str(float(left)/(10**float(val2))) + right
            val1 = str(np.round(float(val1), n))
            return(float(val1[:n+2] + 'e' + val2))
        else:
            val2 = str(len(val) - 1)
            if '-' in val: val2 = str(int(val2) - 1)
            val1 = str(float(val)/(10**float(val2)))
            val1 = str(np.round(float(val1), n))
            if len(val1) < n: val1 += '0'*(n - len(val1))
            return(float(val1[:n+2] + 'e' + val2))
    else:
        val1, val2 = val.split('e')
        val1 = str(np.round(float(val1), n))
        if len(val1) < n: val1 += '0'*(n - len(val1))
        return(float(val1[:n+2] + 'e' + val2))

def get_hashes(c, a, b):
    global debug_in, debug_out, i
    if debug_out != None or debug_in != None:
        i += 1
        c2 = np.round(COPY(c).astype(np.float32), sigfigs - 2)
        a2 = np.round(COPY(a).astype(np.float32), sigfigs - 2)
        b2 = np.round(COPY(b).astype(np.float32), sigfigs - 2)
        # The hash function determines that -0 != 0.
        # This changes any instances of -0 to 0.
        c2[c2 == 0] = 0
        b2[b2 == 0] = 0
        a2[a2 == 0] = 0
        # This prevents rounding errors in the max value from influencing results
        for v in np.unique(c2):
            if np.abs(v) >= 10: c2[c2 == v] = round_figures(v, sigfigs - 2)
        for v in np.unique(b2):
            if np.abs(v) >= 10: b2[b2 == v] = round_figures(v, sigfigs - 2)
        for v in np.unique(a2):
            if np.abs(v) >= 10: a2[a2 == v] = round_figures(v, sigfigs - 2)
        hash1, hash2 = hash(np.concatenate([a2,b2])), hash(c2)
        if debug_out != None:
            global debug_vals
            debug_vals.write(hash1 + "\t" +  hash2 + "\t" + str(i) + "\n")
        if debug_in != None:
            global input_hash_log
            row = input_hash_log.loc[input_hash_log["index"] == str(i), ["hash1", "hash2"]]
            old_hash1, old_hash2 = row.to_numpy().reshape(-1)
            if hash1 != old_hash1 or hash2 != old_hash2:
                pdb.set_trace()

def saferound(a, n):
    a2 = COPY(a)
    if type(a2) == np.ndarray:
        if type(a2[0]) == np.bool_:
            return(a2.astype(np.int8))
        else:
            return(np.round(a2, n))
    else:
        return(np.round(a2, n))

def bound(c):
    if type(c[0]) == np.float64 or type(c[0]) == np.int64:
        pdb.set_trace()
    c2 = COPY(c)
    c2[c2 > MAX] = MAX
    c2[c2 < -MAX] = -MAX
    if type(c2[0]) == np.float32:
        c2 = saferound(c2, sigfigs)
    return(c2)

def modulus(a, b):
    #pdb.set_trace()
    """ if b != 0 return absolute value of (a) % b else return 1 """
    name = "modulus"
    a2, b2 = COPY(a), COPY(b)
    if type(a[0]) == np.bool_: a2 = a2.astype(np.int8)
    if type(b[0]) == np.bool_: b2 = b2.astype(np.int8)
    zeros = (saferound(b, sigfigs) == 0)
    a2[zeros] = 1
    b2[zeros] = 2
    a3, b3 = np.abs(a2), np.abs(b2)
    c = saferound(a3, sigfigs - 2) % saferound(b2, sigfigs - 2)
    mag_diff_too_big = np.log10(a3/np.max([b3, a3/1E12]) + 1) > 10
    c[mag_diff_too_big] = 0
    get_hashes(c, a, b)
    return(bound(c))

#----------------------------------#
def safediv(a, b):
    #pdb.set_trace()
    name = "safediv"
    """ a divided by b if b != 0 else returns 1 """
    a2, b2 = COPY(a), COPY(b)
    zeros = (b == 0)
    b2[zeros] = 1
    a2[zeros] = 1
    c = np.divide(a2, b2, dtype = np.float32)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def plus_mod_two(a, b):
    #pdb.set_trace()
    name = "plus_mod_2"
    """ take absolute value of a + b and mod result with 2 """
    a2, b2 = COPY(a), COPY(b)
    if type(a[0]) == np.bool_: a2 = a2.astype(np.int8)
    if type(b[0]) == np.bool_: b2 = b2.astype(np.int8)
    c = saferound(np.abs(a2 + b2), sigfigs - 2) % 2
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def addition(a, b):
    #pdb.set_trace()
    """ return sum of a and b """
    name = "addition"
    a2, b2 = COPY(a), COPY(b)
    if type(a[0]) == np.bool_: a2 = a2.astype(np.int8)
    if type(b[0]) == np.bool_: b2 = b2.astype(np.int8)
    c = a2 + b2
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def subtract(a, b):
    #pdb.set_trace()
    """ returns the difference between a and b """
    name = "subtract"
    a2, b2 = COPY(a), COPY(b)
    if type(a[0]) == np.bool_: a2 = a2.astype(np.int8)
    if type(b[0]) == np.bool_: b2 = b2.astype(np.int8)
    c = a2 - b2
    c[c < -MAX] = -MAX
    c[c > MAX] = MAX    
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def multiply(a, b):
    #pdb.set_trace()
    """ returns the multiple of a and b """
    name = "multiply"
    a2, b2 = COPY(a), COPY(b)
    c = a2*b2
    c[c < -MAX] = -MAX
    c[c > MAX] = MAX
    get_hashes(c, a, b)
    return(bound(c))

###################### LOGIC OPERATORS #################################

def not_equal(a, b):
    #pdb.set_trace()
    """ return 1 if True, else 0 """
    name = "not_equal"
    a2, b2 = COPY(a), COPY(b)
    c = saferound(a2, sigfigs) != saferound(b2, sigfigs)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def equal(a, b):
    #pdb.set_trace()
    """ return 1 if True, else 0 """
    name = "equal"
    a2, b2 = COPY(a), COPY(b)
    c = saferound(a2, sigfigs) == saferound(b2, sigfigs)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def lt(a, b):
    #pdb.set_trace()
    """ return 1 if True, else 0 """
    name = "lt"
    a2, b2 = COPY(a), COPY(b)
    c = saferound(a2, sigfigs) < saferound(b2, sigfigs)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def gt(a, b):
    #pdb.set_trace()
    """ return 1 if True, else 0 """
    name = "gt"
    a2, b2 = COPY(a), COPY(b)
    c = saferound(a2, sigfigs) > saferound(b2, sigfigs)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def OR(a, b):
    #pdb.set_trace()
    """ return 1 if True, else 0 """
    name = "OR"
    a2, b2 = COPY(a), COPY(b)
    a_nums = saferound(a2, sigfigs) != 0
    b_nums = saferound(b2, sigfigs) != 0
    c = np.logical_or(a_nums, b_nums)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def xor(a, b):
    #pdb.set_trace()
    """ do xor on values anded with 0 """
    name = "xor"
    a2, b2 = COPY(a), COPY(b)
    a_nums = saferound(a2, sigfigs) != 0
    a_zeros = saferound(a2, sigfigs) == 0
    b_nums = saferound(b2, sigfigs) != 0
    b_zeros = saferound(b2, sigfigs) == 0
    a_not_b = np.logical_and(a_nums, b_zeros)
    b_not_a = np.logical_and(a_zeros, b_nums)
    c = np.logical_or(a_not_b, b_not_a)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def AND(a, b):
    #pdb.set_trace()
    """ logical and of a and b """
    name = "AND"
    a2, b2 = COPY(a), COPY(b)
    a_nums = saferound(a2, sigfigs) != 0
    b_nums = saferound(b2, sigfigs) != 0
    #pdb.set_trace()
    c = np.logical_and(a_nums, b_nums)
    get_hashes(c, a, b)
    return(bound(c))
####################### BITWISE OPERATORS ##############################
def bitand(a, b):
    #pdb.set_trace()
    name = "bitand"
    a2, b2 = COPY(a), COPY(b)
    if type(a2[0]) == np.float32 or type(b2[0]) == np.float32:
        int_a = saferound(a2, 0).astype(int)
        int_b = saferound(b2, 0).astype(int)
        c = np.bitwise_and(int_a, int_b).astype(np.float32)
    else:
        if type(a2[0]) == np.bool_: a2 = a2.astype(np.int8)
        if type(b2[0]) == np.bool_: b2 = b2.astype(np.int8)
        c = np.bitwise_and(a2, b2, dtype = np.int8)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def bitor(a, b):
    #pdb.set_trace()
    """ bitwise OR: 110 | 101 eq 111 """
    name = "bitor"
    a2, b2 = COPY(a), COPY(b)
    if type(a2[0]) == np.float32 or type(b2[0]) == np.float32:
        int_a = saferound(a2, 0).astype(int)
        int_b = saferound(b2, 0).astype(int)
        c = np.bitwise_or(int_a, int_b).astype(np.float32)
    else:
        if type(a2[0]) == np.bool_: a2 = a2.astype(np.int8)
        if type(b2[0]) == np.bool_: b2 = b2.astype(np.int8)
        c = np.bitwise_or(a2, b2, dtype = np.int8)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def bitxor(a, b):
    #pdb.set_trace()
    """ bitwise XOR: 110 ^ 101 eq 011 """
    name = "bitxor"
    a2, b2 = COPY(a), COPY(b)
    if type(a2[0]) == np.float32 or type(b2[0]) == np.float32:
        int_a = saferound(a2, 0).astype(int)
        int_b = saferound(b2, 0).astype(int)
        c = np.bitwise_xor(int_a, int_b).astype(np.float32)
    else:
        if type(a2[0]) == np.bool_: a2 = a2.astype(np.int8)
        if type(b2[0]) == np.bool_: b2 = b2.astype(np.int8)
        c = np.bitwise_xor(a2, b2, dtype = np.int8)
    get_hashes(c, a, b)
    return(bound(c))

######################## UNARY OPERATORS ###############################
def ABS(a):
    """ return absolute value """
    name = "ABS"
    a2 = COPY(a)
    c = np.abs(a2)
    get_hashes(c, a, a)
    return(bound(c))
#----------------------------------#
def factorial(f_in):
    #pdb.set_trace()
    name = "factorial"
    f = np.abs(saferound(f_in, 0))
    FACTMAX_indices = f > FACTMAX
    f[FACTMAX_indices] = FACTMAX

    f_max = int(np.max(f))
    f_matrix = np.ones((len(f), f_max), dtype = type(f[0]))
    f_matrix = f.reshape(-1,1)*f_matrix
    f_matrix -= np.arange(0, np.max(f), dtype = type(f[0]))
    f_matrix[f_matrix < 1] = 1

    if f_max < 6:
        c = np.product(f_matrix, axis = 1, dtype = np.int8)
    else:
        c = np.product(f_matrix, axis = 1, dtype = np.float32)
        c[FACTMAX_indices] = MAX

    get_hashes(c, f_in, f_in)
    return(bound(c))
#----------------------------------#
def NOT(a):
    #pdb.set_trace()
    name = "NOT"
    a2 = COPY(a)
    c = saferound(a2, sigfigs) == 0 
    get_hashes(c, a, a)
    return(bound(c))
#----------------------------------#
def log10ofA(a):
    #pdb.set_trace()
    """ Return the logarithm of a to base 10. """
    name = "log10ofA"
    a2 = saferound(a, sigfigs)
    a2[a2 == 0] = 1
    a2[a2 < 0] = np.abs(a2[a2 < 0])
    c = np.log10(a2.astype(np.float32))
    get_hashes(c, a, a)
    return(bound(c))
#----------------------------------#
def log2ofA(a):
    #pdb.set_trace()
    """ Return the logarithm of a to base 2. """
    name = "log2ofA"
    a2 = saferound(a, sigfigs)
    a2[a2 == 0] = 1
    a2[a2 < 0] = np.abs(a2[a2 < 0])
    c = np.log2(a2.astype(np.float32))
    get_hashes(c, a, a)
    return(bound(c))
#----------------------------------#
def logEofA(a):
    #pdb.set_trace()
    """ Return the natural logarithm of a. """
    name = "logEofA"
    a2 = saferound(a, sigfigs)
    a2[a2 == 0] = 1
    a2[a2 < 0] = np.abs(a2[a2 < 0])
    c = np.log(a2.astype(np.float32))
    get_hashes(c, a, a)
    return(bound(c))

######################## LARGE OPERATORS ###############################
def power(a, b):
    #pdb.set_trace()
    """ return a to the power of b or MAX, whichever is less. """
    name = "power"
    a2 = COPY(a)
    b2 = COPY(b)
    b2[b2 > 100] = 100
    
    global debug_in, debug_out
    debug_mode = debug_in != None or debug_out != None
    if debug_mode:
        a_abs = np.abs(a2).astype(np.float64)  # ensure the denial of complex number creation
    else:
        a_abs = np.abs(a2).astype(np.float32)
    zeros = saferound(a_abs, sigfigs) == 0
    ones = np.logical_and(zeros, saferound(b2, sigfigs) == 0)
    div_by_0 = np.logical_and(zeros, b2 < 0)
    b2[div_by_0] = 1
    a_abs[zeros] = 0.1

    if debug_mode:
        logc = b2.astype(np.float64)*np.log10(a_abs)
    else:
        logc = b2.astype(np.float32)*np.log10(a_abs)
    large_vals = logc > np.log10(MAX)
    logc[large_vals] = 1
    c = np.power(10, logc)
    if type(c[0]) == np.float64:
        c = c.astype(np.float32) 
    c[zeros] = 0
    c[large_vals] = MAX
    c[div_by_0] = MAX
    c[ones] = 1
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def logAofB(a, b):
    #pdb.set_trace()
    """ Return the logarithm of a to the given b. """
    name = "logAofB"
    a2 = saferound(a, sigfigs)
    a2[a2 == 0] = 1
    a2[a2 < 0] = np.abs(a2[a2 < 0])
    b2 = saferound(b, sigfigs)
    b2[b2 == 0] = 1
    b2[b2 < 0] = np.abs(b2[b2 < 0])

    alog = np.log(a2.astype(np.float32))
    blog = np.log(b2.astype(np.float32))
    zeros = (saferound(blog, sigfigs) == 0)
    blog[zeros] = 1
    alog[zeros] = 1
    c = np.divide(alog, blog, dtype = np.float32)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------# 
def permute(a, b):
    #pdb.set_trace()
    """ reordering elements """
    name = "permute"
    a2, b2 = COPY(a), COPY(b)
    a2 = np.abs(saferound(a2, 0))
    b2 = np.abs(saferound(b2, 0))
    a2[a2 > FACTMAX] = FACTMAX
    b2[b2 > FACTMAX] = FACTMAX
    indices = b2 > a2
    a_new_vals = b2[indices]
    b_new_vals = a2[indices]
    if type(a2[0]) == np.int8: 
        if type(b2[0]) == np.float32: a2 = a2.astype(np.float32)
    if type(b2[0]) == np.int8: 
        if type(a2[0]) == np.float32: b2 = b2.astype(np.float32)
    b2[indices] = b_new_vals
    a2[indices] = a_new_vals
    
    # If a = MAX and b2 = 0, then the operator = 1
    # Else, the operator >= MAX, which gets bounded at MAX
    a2[a2 > FACTMAX] = FACTMAX
    c = np.divide(factorial(a2), factorial(a2 - b2), dtype = np.float32) 
    c[np.logical_and(a2 == FACTMAX, b2 > 0)] = MAX
    c[np.logical_and(a2 == FACTMAX, b2 == 0)] = 1
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def choose(a, b):
    #pdb.set_trace()
    """ n Choose r function """
    name = "choose"
    a2, b2 = COPY(a), COPY(b)
    a2 = np.abs(saferound(a2, 0))
    b2 = np.abs(saferound(b2, 0))
    indices = b2 > a2
    a_new_vals = b2[indices]
    b_new_vals = a2[indices]
    if type(a2[0]) == np.int8: 
        if type(b2[0]) == np.float32: a2 = a2.astype(np.float32)
    if type(b2[0]) == np.int8: 
        if type(a2[0]) == np.float32: b2 = b2.astype(np.float32)
    b2[indices] = b_new_vals
    a2[indices] = a_new_vals

    # If a = MAX and b2 = 0, then the operator = 1
    # Else, the operator >= MAX, which gets bounded at MAX
    equal_indices = (a2 == b2)
    a2[a2 > FACTMAX] = FACTMAX
    numerator = factorial(a2).astype(np.float32)
    d1, d2 = factorial(b2).astype(np.float32), factorial(a2 - b2).astype(np.float32)
    c = np.divide(numerator, d1*d2)
    c[np.logical_and(a2 == FACTMAX, b2 > 0)] = MAX
    c[np.logical_and(a2 == FACTMAX, b2 == 0)] = 1
    c[equal_indices] = 1 
    get_hashes(c, a, b)
    return(bound(c))
######################### MISC OPERATORS ###############################

def minimum(a, b):
    #pdb.set_trace()
    """ return the smallest value of a and b """
    name = "minimum"
    a2, b2 = COPY(a), COPY(b)
    c = np.min([a2, b2], axis = 0)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def maximum(a, b):
    #pdb.set_trace()
    """ return the largest value of a and b """
    name = "maximum"
    a2, b2 = COPY(a), COPY(b)
    c = np.max([a2, b2], axis = 0)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def left(a, b):
    #pdb.set_trace()
    """ return left value """
    name = "left"
    get_hashes(a, a, b)
    return(bound(a))
#----------------------------------#
def right(a, b):
    #pdb.set_trace()
    """ return right value """
    name = "right"
    get_hashes(a, a, b)
    return(bound(b))
