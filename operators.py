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
from copy import deepcopy as COPY
#largest factorial less than numpy's 64 bit integer maximum 
FACTMAX = 20
MAX = np.product(FACTMAX  - np.arange(0, FACTMAX, dtype = np.int64))
import pdb

###################### BASIC OPERATORS #################################

def bound(c):
    c[c > MAX] = MAX
    c[c < -MAX] = -MAX
    return(c)

def create_no_inf_copy(*args):
    args2 = []
    for arg in args:
        arg2 = COPY(arg)
        arg2[arg2 > MAX] = MAX
        arg2[arg2 < -MAX] = -MAX
        if len(args) == 1:
            return(arg2)
        args2.append(arg2)
    return(args2)

def modulus(a,b):
    """ if b != 0 return absolute value of (a) % b
        else return 1 """
    a2, b2 = create_no_inf_copy(a, b)
    zeros = (np.round(b, 8) == 0)
    a2[zeros] = 1
    b2[zeros] = 2  
    c = np.abs(a2) % b2
    return(bound(c))

#----------------------------------#
def safediv(a,b):
    """ a divided by b if b != 0 else
        returns 1 """
    a2, b2 = create_no_inf_copy(a, b)
    zeros = (b == 0)
    b2[zeros] = 1
    a2[zeros] = 1
    c = a2/b2
    return(bound(c))
#----------------------------------#
def plus_mod_two(a,b):
    """ take absolute value of a + b
        and mod result with 2 """
    a2, b2 = create_no_inf_copy(a, b)
    c = np.abs(a2 + b2) % 2
    return(bound(c))
#----------------------------------#
def addition(a,b):
    """ return sum of a and b """
    a2, b2 = create_no_inf_copy(a, b)
    c = a2 + b2
    return(bound(c))
#----------------------------------#
def subtract(a,b):
    """ returns the difference between
        a and b """
    a2, b2 = create_no_inf_copy(a, b)
    c = a2 - b2
    c[c < -MAX] = -MAX
    c[c > MAX] = MAX
    return(bound(c))
#----------------------------------#
def multiply(a,b):
    """ returns the multiple of a and b """
    a2, b2 = create_no_inf_copy(a, b)
    c = a2*b2
    c[c < -MAX] = -MAX
    c[c > MAX] = MAX
    return(bound(c))

###################### LOGIC OPERATORS #################################

def not_equal(a,b):
    """ return 1 if True, else 0 """
    a2, b2 = create_no_inf_copy(a, b)
    c = np.round(a2, 8) != np.round(b2, 8)
    return(bound(c).astype(np.int64))
#----------------------------------#
def equal(a,b):
    """ return 1 if True, else 0 """
    a2, b2 = create_no_inf_copy(a, b)
    c = np.round(a2, 8) == np.round(b2, 8)
    return(bound(c).astype(np.int64))
#----------------------------------#
def lt(a,b):
    """ return 1 if True, else 0 """
    a2, b2 = create_no_inf_copy(a, b)
    c = (np.round(a2, 8) < np.round(b2, 8))
    return(bound(c).astype(np.int64))
#----------------------------------#
def gt(a,b):
    """ return 1 if True, else 0 """
    a2, b2 = create_no_inf_copy(a, b)
    c = (np.round(a2, 8) > np.round(b2, 8))
    return(bound(c).astype(np.int64))
#----------------------------------#
def OR(a,b):
    """ return 1 if True, else 0 """
    a2, b2 = create_no_inf_copy(a, b)
    a_nums = np.round(a2, 8) != 0
    b_nums = np.round(b2, 8) != 0
    c = np.logical_or(a_nums, b_nums)
    return(bound(c).astype(np.int64))
#----------------------------------#
def xor(a,b):
    """ do xor on values anded with 0 """
    a2, b2 = create_no_inf_copy(a, b)
    a_nums = np.round(a2, 8) != 0
    a_zeros = np.round(a2, 8) == 0
    b_nums = np.round(b2, 8) != 0
    b_zeros = np.round(b2, 8) == 0
    a_not_b = np.logical_and(a_nums, b_zeros)
    b_not_a = np.logical_and(a_zeros, b_nums)
    c = np.logical_or(a_not_b, b_not_a)
    return(bound(c).astype(np.int64))
#----------------------------------#
def AND(a,b):
    """ logical and of a and b """
    a2, b2 = create_no_inf_copy(a, b)
    a_nums = np.round(a2, 8) != 0
    b_nums = np.round(b2, 8) != 0
    c = np.logical_and(a_nums, b_nums)
    return(bound(c).astype(np.int64))
####################### BITWISE OPERATORS ##############################
def bitand(a,b):
    a2, b2 = create_no_inf_copy(a, b)
    int_a = a2.astype(np.int64)
    int_b = b2.astype(np.int64)
    c = np.bitwise_and(int_a, int_b)
    return(bound(c).astype(np.int64))
#----------------------------------#
def bitor(a,b):
    """ bitwise OR: 110 | 101 eq 111 """
    a2, b2 = create_no_inf_copy(a, b)
    int_a = a2.astype(np.int64)
    int_b = b2.astype(np.int64)
    c = np.bitwise_or(int_a, int_b)
    return(bound(c).astype(np.int64))
#----------------------------------#
def bitxor(a,b):
    """ bitwise XOR: 110 ^ 101 eq 011 """
    a2, b2 = create_no_inf_copy(a, b)
    int_a = a2.astype(np.int64)
    int_b = b2.astype(np.int64)
    c = np.bitwise_xor(int_a, int_b)
    return(bound(c).astype(np.int64))

######################## UNARY OPERATORS ###############################
def ABS(a):
    """ return absolute value """
    a2 = create_no_inf_copy(a)
    c = np.abs(a2)
    return(bound(c))
#----------------------------------#
def factorial(f_in):
    f = np.abs(np.round(f_in, 0))
    f[f > FACTMAX] = FACTMAX
    f = f.astype(np.int64)
    f_matrix = np.ones((len(f), np.max(f)))
    f_matrix = f.reshape(-1,1)*f_matrix
    f_matrix -= np.arange(0, np.max(f))
    f_matrix[f_matrix < 1] = 1
    c = np.product(f_matrix, axis = 1)
    return(bound(c))
#----------------------------------#
def NOT(a):
    a2 = create_no_inf_copy(a)
    c = np.round(a2, 8) == 0 
    return(bound(c).astype(np.int64))
#----------------------------------#
def log10ofA(a):
    """ Return the logarithm of a to base 10. """
    inf_indices = np.logical_or(a == np.inf, a == -np.inf)
    a2 = create_no_inf_copy(a)
    a2[np.round(a2, 8) == 0] = 1
    a2[np.round(a2, 8) < 0] = np.abs(a2[np.round(a2, 8) < 0])
    c = np.log10(a2)
    c[inf_indices] = MAX
    return(bound(c))
#----------------------------------#
def log2ofA(a):
    """ Return the logarithm of a to base 2. """
    inf_indices = np.logical_or(a == np.inf, a == -np.inf)
    a2 = create_no_inf_copy(a)
    a2[np.round(a2, 8) == 0] = 1
    a2[np.round(a2, 8) < 0] = np.abs(a2[np.round(a2, 8) < 0])
    c = np.log2(a2)
    c[inf_indices] = MAX
    return(bound(c))
#----------------------------------#
def logEofA(a):
    """ Return the natural logarithm of a. """
    inf_indices = np.logical_or(a == np.inf, a == -np.inf)
    a2 = create_no_inf_copy(a)
    a2[np.round(a2, 8) == 0] = 1
    a2[np.round(a2, 8) < 0] = np.abs(a2[np.round(a2, 8) < 0])
    c = np.log(a2)
    c[inf_indices] = MAX
    return(bound(c))

######################## LARGE OPERATORS ###############################
def power(a,b):

    """ return a to the power of b or MAX, whichever is less. """
    a2 = create_no_inf_copy(a)
    b2 = COPY(b)
    b2[b2 > 100] = 100
    a_abs = np.abs(a2)  # ensure the denial of complex number creation
    zeros = np.round(a_abs, 8) == 0
    ones = np.logical_and(zeros, np.round(b2, 8) == 0)
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
    return(bound(c))
#----------------------------------#
def logAofB(a,b):
    """ Return the logarithm of a to the given b. """
    a2, b2 = create_no_inf_copy(a, b)
    a2[np.round(a2, 8) == 0] = 1
    a2[np.round(a2, 8) < 0] = np.abs(a2[np.round(a2, 8) < 0])
    b2[np.round(b2, 8) == 0] = 1
    b2[np.round(b2, 8) < 0] = np.abs(b2[np.round(b2, 8) < 0])

    alog = np.log(a2)
    blog = np.log(b2)
    zeros = (np.round(blog, 8) == 0)
    blog[zeros] = 1
    alog[zeros] = 1
    c = alog/blog
    
    return(bound(c))
#----------------------------------# 
def permute(a,b):
    """ reordering elements """
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
    c = factorial(a2)/(factorial(a2 - b2))
    c[np.logical_and(a2 == FACTMAX, b2 > 0)] = MAX
    c[np.logical_and(a2 == FACTMAX, b2 == 0)] = 1
    return(bound(c))
#----------------------------------#
def choose(a,b):
    """ n Choose r function """
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
    c = factorial(a2)/(factorial(b2)*factorial(a2 - b2))
    c[np.logical_and(a2 == FACTMAX, b2 > 0)] = MAX
    c[np.logical_and(a2 == FACTMAX, b2 == 0)] = 1
    c[equal_indices] = 1
    return(bound(c))
######################### MISC OPERATORS ###############################

def minimum(a,b):
    """ return the smallest value of a and b """
    a2, b2 = create_no_inf_copy(a, b)
    c = np.min([a2, b2], axis = 0)
    return(bound(c))
#----------------------------------#
def maximum(a,b):
    """ return the largest value of a and b """
    a2, b2 = create_no_inf_copy(a, b)
    c = np.max([a2, b2], axis = 0)
    return(bound(c))
#----------------------------------#
def left(a, b):
    """ return left value """
    return(bound(a))
#----------------------------------#
def right(a, b):
    """ return right value """
    return(bound(b))
