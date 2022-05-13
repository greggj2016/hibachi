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

def bound(c):
    c2 = COPY(c).astype(np.float64)
    c2[c2 > MAX] = MAX
    c2[c2 < -MAX] = -MAX
    return(c2)

def create_no_inf_copy(*args):
    args2 = []
    for arg in args:
        arg2 = COPY(arg).astype(np.float64)
        arg2[arg2 > MAX] = MAX
        arg2[arg2 < -MAX] = -MAX
        if len(args) == 1:
            return(arg2)
        args2.append(arg2)
    return(args2)

def modulus(a,b):
    #pdb.set_trace()
    """ if b != 0 return absolute value of (a) % b else return 1 """
    name = "modulus"
    a2, b2 = create_no_inf_copy(a, b)
    zeros = (np.round(b, sigfigs) == 0)
    a2[zeros] = 1
    b2[zeros] = 2
    a3, b3 = np.abs(a2), np.abs(b2)
    c = np.round(a3, sigfigs - 2) % np.round(b2, sigfigs - 2)
    mag_diff_too_big = np.log10(a3/np.max([b3, a3/1E12]) + 1) > 10
    c[mag_diff_too_big] = 0
    get_hashes(c, a, b)
    return(bound(c))

#----------------------------------#
def safediv(a,b):
    #pdb.set_trace()
    """ a divided by b if b != 0 else returns 1 """
    name = "safediv"
    a2, b2 = create_no_inf_copy(a, b)
    zeros = (b == 0)
    b2[zeros] = 1
    a2[zeros] = 1
    c = a2/b2
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def plus_mod_two(a,b):
    #pdb.set_trace()
    """ take absolute value of a + b and mod result with 2 """
    name = "plus_mod_two"
    a2, b2 = create_no_inf_copy(a, b)
    c = np.round(np.abs(a2 + b2), sigfigs - 2) % 2
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def addition(a,b):
    #pdb.set_trace()
    """ return sum of a and b """
    name = "addition"
    a2, b2 = create_no_inf_copy(a, b)
    c = a2 + b2
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def subtract(a,b):
    #pdb.set_trace()
    """ returns the difference between
        a and b """
    name = "subtract"
    a2, b2 = create_no_inf_copy(a, b)
    c = a2 - b2
    c[c < -MAX] = -MAX
    c[c > MAX] = MAX
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def multiply(a,b):
    #pdb.set_trace()
    """ returns the multiple of a and b """
    name = "multiply"
    a2, b2 = create_no_inf_copy(a, b)
    c = a2*b2
    c[c < -MAX] = -MAX
    c[c > MAX] = MAX
    get_hashes(c, a, b)
    return(bound(c))

###################### LOGIC OPERATORS #################################

def not_equal(a,b):
    #pdb.set_trace()
    """ return 1 if True, else 0 """
    name = "not_equal"
    a2, b2 = create_no_inf_copy(a, b)
    c = np.round(a2, sigfigs) != np.round(b2, sigfigs)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def equal(a,b):
    #pdb.set_trace()
    """ return 1 if True, else 0 """
    name = "equal"
    a2, b2 = create_no_inf_copy(a, b)
    c = np.round(a2, sigfigs) == np.round(b2, sigfigs)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def lt(a,b):
    #pdb.set_trace()
    """ return 1 if True, else 0 """
    name = "lt"
    a2, b2 = create_no_inf_copy(a, b)
    c = (np.round(a2, sigfigs) < np.round(b2, sigfigs))
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def gt(a,b):
    #pdb.set_trace()
    """ return 1 if True, else 0 """
    name = "gt"
    a2, b2 = create_no_inf_copy(a, b)
    c = (np.round(a2, sigfigs) > np.round(b2, sigfigs))
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def OR(a,b):
    #pdb.set_trace()
    """ return 1 if True, else 0 """
    name = "OR"
    a2, b2 = create_no_inf_copy(a, b)
    a_nums = np.round(a2, sigfigs) != 0
    b_nums = np.round(b2, sigfigs) != 0
    c = np.logical_or(a_nums, b_nums)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def xor(a,b):
    #pdb.set_trace()
    """ do xor on values anded with 0 """
    name = "xor"
    a2, b2 = create_no_inf_copy(a, b)
    a_nums = np.round(a2, sigfigs) != 0
    a_zeros = np.round(a2, sigfigs) == 0
    b_nums = np.round(b2, sigfigs) != 0
    b_zeros = np.round(b2, sigfigs) == 0
    a_not_b = np.logical_and(a_nums, b_zeros)
    b_not_a = np.logical_and(a_zeros, b_nums)
    c = np.logical_or(a_not_b, b_not_a)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def AND(a,b):
    #pdb.set_trace()
    """ logical and of a and b """
    name = "AND"
    a2, b2 = create_no_inf_copy(a, b)
    a_nums = np.round(a2, sigfigs) != 0
    b_nums = np.round(b2, sigfigs) != 0
    c = np.logical_and(a_nums, b_nums)
    get_hashes(c, a, b)
    return(bound(c))
####################### BITWISE OPERATORS ##############################
def bitand(a,b):
    #pdb.set_trace()
    name = "bitand"
    a2, b2 = create_no_inf_copy(a, b)
    int_a = np.round(a2, 0).astype(np.int64)
    int_b = np.round(b2, 0).astype(np.int64)
    c = np.bitwise_and(int_a, int_b)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def bitor(a,b):
    #pdb.set_trace()
    """ bitwise OR: 110 | 101 eq 111 """
    name = "bitor"
    a2, b2 = create_no_inf_copy(a, b)
    int_a = np.round(a2, 0).astype(np.int64)
    int_b = np.round(b2, 0).astype(np.int64)
    c = np.bitwise_or(int_a, int_b)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def bitxor(a,b):
    #pdb.set_trace()
    """ bitwise XOR: 110 ^ 101 eq 011 """
    name = "bitxor"
    a2, b2 = create_no_inf_copy(a, b)
    int_a = np.round(a2, 0).astype(np.int64)
    int_b = np.round(b2, 0).astype(np.int64)
    c = np.bitwise_xor(int_a, int_b)
    get_hashes(c, a, b)
    return(bound(c))

######################## UNARY OPERATORS ###############################
def ABS(a):
    """ return absolute value """
    name = "ABS"
    a2 = create_no_inf_copy(a)
    c = np.abs(a2)
    get_hashes(c, a, a)
    return(bound(c))
#----------------------------------#
def factorial(f_in):
    #pdb.set_trace()
    name = "factorial"
    f = COPY(f_in)
    f = np.abs(np.round(f, 0))
    FACTMAX_indices = f > FACTMAX
    f[FACTMAX_indices] = FACTMAX
    f = np.round(f, 0).astype(np.int64)
    f_matrix = np.ones((len(f), np.round(np.max(f), 0)))
    f_matrix = f.reshape(-1,1)*f_matrix
    f_matrix -= np.arange(0, np.max(f))
    f_matrix[f_matrix < 1] = 1
    c = np.product(f_matrix, axis = 1).astype(np.float64)
    c[FACTMAX_indices] = MAX
    get_hashes(c, f_in, f_in)
    return(bound(c))
#----------------------------------#
def NOT(a):
    #pdb.set_trace()
    name = "NOT"
    a2 = create_no_inf_copy(a)
    c = np.round(a2, sigfigs) == 0 
    get_hashes(c, a, a)
    return(bound(c))
#----------------------------------#
def log10ofA(a):
    #pdb.set_trace()
    """ Return the logarithm of a to base 10. """
    name = "log10ofA"
    inf_indices = np.logical_or(a == np.inf, a == -np.inf)
    a2 = create_no_inf_copy(a)
    a2[np.round(a2, sigfigs) == 0] = 1
    a2[np.round(a2, sigfigs) < 0] = np.abs(a2[np.round(a2, sigfigs) < 0])
    c = np.log10(a2)
    c[inf_indices] = MAX
    get_hashes(c, a, a)
    return(bound(c))
#----------------------------------#
def log2ofA(a):
    #pdb.set_trace()
    """ Return the logarithm of a to base 2. """
    name = "log2ofA"
    inf_indices = np.logical_or(a == np.inf, a == -np.inf)
    a2 = create_no_inf_copy(a)
    a2[np.round(a2, sigfigs) == 0] = 1
    a2[np.round(a2, sigfigs) < 0] = np.abs(a2[np.round(a2, sigfigs) < 0])
    c = np.log2(a2)
    c[inf_indices] = MAX
    get_hashes(c, a, a)
    return(bound(c))
#----------------------------------#
def logEofA(a):
    #pdb.set_trace()
    """ Return the natural logarithm of a. """
    name = "logEofA"
    inf_indices = np.logical_or(a == np.inf, a == -np.inf)
    a2 = create_no_inf_copy(a)
    a2[np.round(a2, sigfigs) == 0] = 1
    a2[np.round(a2, sigfigs) < 0] = np.abs(a2[np.round(a2, sigfigs) < 0])
    c = np.log(a2)
    c[inf_indices] = MAX
    get_hashes(c, a, a)
    return(bound(c))

######################## LARGE OPERATORS ###############################
def power(a,b):
    #pdb.set_trace()
    """ return a to the power of b or MAX, whichever is less. """
    name = "power"
    a2 = create_no_inf_copy(a)
    b2 = COPY(b)
    b2[b2 > 100] = 100
    a_abs = np.abs(a2).astype(float)  # ensure the denial of complex number creation
    zeros = np.round(a_abs, sigfigs) == 0
    ones = np.logical_and(zeros, np.round(b2, sigfigs) == 0)
    div_by_0 = np.logical_and(zeros, b2 < 0)
    b2[div_by_0] = 1
    a_abs[zeros] = 0.1

    logc = b2.astype(float)*np.log10(a_abs)
    large_vals = logc > np.log10(MAX)
    logc[large_vals] = 1
    c = 10**logc
    c[zeros] = 0
    c[large_vals] = MAX
    c[div_by_0] = MAX
    c[ones] = 1
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def logAofB(a,b):
    #pdb.set_trace()
    """ Return the logarithm of a to the given b. """
    name = "logAofB"
    a2, b2 = create_no_inf_copy(a, b)
    a2[np.round(a2, sigfigs) == 0] = 1
    a2[np.round(a2, sigfigs) < 0] = np.abs(a2[np.round(a2, sigfigs) < 0])
    b2[np.round(b2, sigfigs) == 0] = 1
    b2[np.round(b2, sigfigs) < 0] = np.abs(b2[np.round(b2, sigfigs) < 0])

    alog = np.log(a2)
    blog = np.log(b2)
    zeros = (np.round(blog, sigfigs) == 0)
    blog[zeros] = 1
    alog[zeros] = 1
    c = alog/blog
    
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------# 
def permute(a,b):
    #pdb.set_trace()
    """ reordering elements """
    name = "permute"
    a2, b2 = create_no_inf_copy(a, b)
    a2 = np.abs(np.round(a2, 0)).astype(np.int64)
    b2 = np.abs(np.round(b2, 0)).astype(np.int64)
    a2[a2 > FACTMAX] = FACTMAX
    b2[b2 > FACTMAX] = FACTMAX
    indices = b2 > a2
    a_new_vals = b2[indices]
    b_new_vals = a2[indices]
    b2[indices] = b_new_vals
    a2[indices] = a_new_vals
    
    # If a = MAX and b2 = 0, then the operator = 1
    # Else, the operator >= MAX, which gets bounded at MAX
    a2[a2 > FACTMAX] = FACTMAX
    c = (factorial(a2)/(factorial(a2 - b2))).astype(np.float64)
    c[np.logical_and(a2 == FACTMAX, b2 > 0)] = MAX
    c[np.logical_and(a2 == FACTMAX, b2 == 0)] = 1
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def choose(a,b):
    #pdb.set_trace()
    """ n Choose r function """
    name = "choose"
    a2, b2 = create_no_inf_copy(a, b)
    a2 = np.abs(np.round(a2, 0)).astype(np.int64)
    b2 = np.abs(np.round(b2, 0)).astype(np.int64)
    indices = b2 > a2
    a_new_vals = b2[indices]
    b_new_vals = a2[indices]
    b2[indices] = b_new_vals
    a2[indices] = a_new_vals

    # If a = MAX and b2 = 0, then the operator = 1
    # Else, the operator >= MAX, which gets bounded at MAX
    equal_indices = (a2 == b2)
    a2[a2 > FACTMAX] = FACTMAX
    c = (factorial(a2)/(factorial(b2)*factorial(a2 - b2))).astype(np.float64)
    c[np.logical_and(a2 == FACTMAX, b2 > 0)] = MAX
    c[np.logical_and(a2 == FACTMAX, b2 == 0)] = 1
    c[equal_indices] = 1
    get_hashes(c, a, b)
    return(bound(c))
######################### MISC OPERATORS ###############################

def minimum(a,b):
    #pdb.set_trace()
    """ return the smallest value of a and b """
    name = "minimum"
    a2, b2 = create_no_inf_copy(a, b)
    c = np.min([a2, b2], axis = 0)
    get_hashes(c, a, b)
    return(bound(c))
#----------------------------------#
def maximum(a,b):
    #pdb.set_trace()
    """ return the largest value of a and b """
    name = "maximum"
    a2, b2 = create_no_inf_copy(a, b)
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
