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
#largest factorial that can be squared. 
FACTMAX = 96.
MAX = np.product(FACTMAX  - np.arange(0, FACTMAX))
import pdb

###################### BASIC OPERATORS #################################

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
    return(c)
#----------------------------------#
def safediv(a,b):
    """ a divided by b if b != 0 else
        returns 1 """
    a2, b2 = create_no_inf_copy(a, b)
    zeros = (np.round(b, 8) == 0)
    b2[zeros] = 1
    a2[zeros] = 1
    c = a2/b2
    c[np.abs(c) > MAX] = MAX
    c[np.abs(c) < -MAX] = -MAX
    return(c)
#----------------------------------#
def plus_mod_two(a,b):
    """ take absolute value of a + b
        and mod result with 2 """
    a2, b2 = create_no_inf_copy(a, b)
    c = np.abs(a2 + b2) % 2
    return(c)
#----------------------------------#
def addition(a,b):
    """ return sum of a and b """
    a2, b2 = create_no_inf_copy(a, b)
    c = a2 + b2
    c[c < -MAX] = -MAX
    c[c > MAX] = MAX
    return(c)
#----------------------------------#
def subtract(a,b):
    """ returns the difference between
        a and b """
    a2, b2 = create_no_inf_copy(a, b)
    c = a2 - b2
    c[c < -MAX] = -MAX
    c[c > MAX] = MAX
    return(c)
#----------------------------------#
def multiply(a,b):
    """ returns the multiple of a and b """
    a2, b2 = create_no_inf_copy(a, b)
    c = a2*b2
    c[c < -MAX] = -MAX
    c[c > MAX] = MAX
    return(c)

###################### LOGIC OPERATORS #################################

def not_equal(a,b):
    """ return 1 if True, else 0 """
    a2, b2 = create_no_inf_copy(a, b)
    c = np.round(a2, 8) != np.round(b2, 8)
    return(c.astype(int))
#----------------------------------#
def equal(a,b):
    """ return 1 if True, else 0 """
    a2, b2 = create_no_inf_copy(a, b)
    c = np.round(a2, 8) == np.round(b2, 8)
    return(c.astype(int))
#----------------------------------#
def lt(a,b):
    """ return 1 if True, else 0 """
    a2, b2 = create_no_inf_copy(a, b)
    c = (np.round(a2, 8) < np.round(b2, 8))
    return(c.astype(int))
#----------------------------------#
def gt(a,b):
    """ return 1 if True, else 0 """
    a2, b2 = create_no_inf_copy(a, b)
    c = (np.round(a2, 8) > np.round(b2, 8))
    return(c.astype(int))
#----------------------------------#
def OR(a,b):
    """ return 1 if True, else 0 """
    a2, b2 = create_no_inf_copy(a, b)
    a_nums = np.round(a2, 8) != 0
    b_nums = np.round(b2, 8) != 0
    c = np.logical_or(a_nums, b_nums)
    return(c.astype(int))
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
    return(c.astype(int))
#----------------------------------#
def AND(a,b):
    """ logical and of a and b """
    a2, b2 = create_no_inf_copy(a, b)
    a_nums = np.round(a2, 8) != 0
    b_nums = np.round(b2, 8) != 0
    c = np.logical_and(a_nums, b_nums)
    return(c.astype(int))
####################### BITWISE OPERATORS ##############################
def bitand(a,b):
    a2, b2 = create_no_inf_copy(a, b)
    int_a = a2.astype(int)
    int_b = b2.astype(int)
    c = np.bitwise_and(int_a, int_b)
    return(c)
#----------------------------------#
def bitor(a,b):
    """ bitwise OR: 110 | 101 eq 111 """
    a2, b2 = create_no_inf_copy(a, b)
    int_a = a2.astype(int)
    int_b = b2.astype(int)
    c = np.bitwise_or(int_a, int_b)
    return(c)
#----------------------------------#
def bitxor(a,b):
    """ bitwise XOR: 110 ^ 101 eq 011 """
    a2, b2 = create_no_inf_copy(a, b)
    int_a = a2.astype(int)
    int_b = b2.astype(int)
    c = np.bitwise_xor(int_a, int_b)
    return(c)

######################## UNARY OPERATORS ###############################
def ABS(a):
    """ return absolute value """
    a2 = create_no_inf_copy(a)
    c = np.abs(a2)
    return(c)
#----------------------------------#
def factorial(a):
    a2 = COPY(a)
    a2[a2 > FACTMAX] = FACTMAX
    a2[a2 < -FACTMAX] = -FACTMAX
    f = np.abs(np.round(a2, 0).astype(int))
    f_matrix = np.ones((len(f), np.max(f)))
    f_matrix = f.reshape(-1,1)*f_matrix
    f_matrix -= np.arange(0, np.max(f))
    f_matrix[f_matrix < 1] = 1
    c = np.product(f_matrix, axis = 1)
    return(c)
#----------------------------------#
def NOT(a):
    a2 = create_no_inf_copy(a)
    c = np.round(a2, 8) == 0 
    return(c.astype(int))
#----------------------------------#
def log10ofA(a):
    """ Return the logarithm of a to base 10. """
    a2 = create_no_inf_copy(a)
    a2[np.round(a2, 8) == 0] = 1
    a2[np.round(a2, 8) < 0] = np.abs(a2[np.round(a2, 8) < 0])
    c = np.log10(a2)
    return(c)
#----------------------------------#
def log2ofA(a):
    """ Return the logarithm of a to base 2. """
    a2 = create_no_inf_copy(a)
    a2[np.round(a2, 8) == 0] = 1
    a2[np.round(a2, 8) < 0] = np.abs(a2[np.round(a2, 8) < 0])
    c = np.log2(a2)
    return(c)
#----------------------------------#
def logEofA(a):
    """ Return the natural logarithm of a. """
    a2 = create_no_inf_copy(a)
    a2[np.round(a2, 8) == 0] = 1
    a2[np.round(a2, 8) < 0] = np.abs(a2[np.round(a2, 8) < 0])
    c = np.log(a2)
    return(c)

######################## LARGE OPERATORS ###############################
def power(a,b):

    """ return a to the power of b or MAX, whichever is less. """
    a2, b2 = create_no_inf_copy(a, b)
    a_abs = np.round(np.abs(a2), 8)  # ensure the denial of complex number creation
    div_by_0 = np.logical_and(a_abs == 0, np.round(b, 8) < 0)
    b2[div_by_0] = 0

    zeros = (a_abs == 0)
    a_abs[zeros] = 1
    logc = b2.astype(float)*np.log10(a_abs)
    logc[logc > np.log10(MAX)] = np.log10(MAX)
    logc[logc < -np.log10(MAX)] = -np.log10(MAX)
    c = 10**logc
    c[zeros] = 0
    return(c)
#----------------------------------#
def logAofB(a,b):
    """ Return the logarithm of a to the given b. """
    a2, b2 = create_no_inf_copy(a, b)
    a2[np.round(a2, 8) == 0] = 1
    a2[np.round(a2, 8) < 0] = np.abs(a2[np.round(a2, 8) < 0])
    alog = np.round(np.log(a2), 8)
    b2[np.round(b2, 8) == 0] = 1
    b2[np.round(b2, 8) < 0] = np.abs(b2[np.round(b2, 8) < 0])
    blog = np.round(np.log(b2), 8)
    
    zeros = (np.round(blog, 8) == 0)
    blog[zeros] = 1
    alog[zeros] = 1
    c = alog/blog
    
    return(c)
#----------------------------------# 
def permute(a,b):
    """ reordering elements """
    a2, b2 = COPY(a), COPY(b)
    indices = b2 > a2
    a_new_vals = b2[indices]
    b_new_vals = a2[indices]
    b2[indices] = b_new_vals
    a2[indices] = a_new_vals
    c =  factorial(a2)/(factorial(subtract(a2, b2)))
    return(c)
#----------------------------------#
def choose(a,b):
    """ n Choose r function """
    a2, b2 = COPY(a), COPY(b)
    indices = b2 > a2
    a_new_vals = b2[indices]
    b_new_vals = a2[indices]
    b2[indices] = b_new_vals
    a2[indices] = a_new_vals
    c =  factorial(a2)/(factorial(b2)*factorial(subtract(a2, b2)))
    return(c)
######################### MISC OPERATORS ###############################
def minimum(a,b):
    """ return the smallest value of a and b """
    a2, b2 = create_no_inf_copy(a, b)
    c = np.min([a2, b2], axis = 0)
    return(c)
#----------------------------------#
def maximum(a,b):
    """ return the largest value of a and b """
    a2, b2 = create_no_inf_copy(a, b)
    c = np.max([a2, b2], axis = 0)
    return(c)
#----------------------------------#
def left(a, b):
    """ return left value """
    return(a)
#----------------------------------#
def right(a, b):
    """ return right value """
    return(b)
