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
FACTMAX = 170.
MAX =np.product(FACTMAX  - np.arange(0, FACTMAX))

###################### BASIC OPERATORS #################################
def modulus(a,b):
    """ if b != 0 return absolute value of (a) % b
        else return 1 """
    c = np.abs(a) % b
    c[np.abs(c) == np.inf] = 1
    return(c)
#----------------------------------#
def safediv(a,b):
    """ a divided by b if b != 0 else
        returns 1 """
    c = a/b
    c[np.abs(c) == np.inf] = 1
    return(c)
#----------------------------------#
def plus_mod_two(a,b):
    """ take absolute value of a + b
        and mod result with 2 """
    c = abs(a + b)%2
    c[np.abs(c) == np.inf] = 1
    return(c)
#----------------------------------#
def addition(a,b):
    """ return sum of a and b """
    c = a + b
    c[c == -np.inf] = -MAX
    c[c == np.inf] = MAX
    return(c)
#----------------------------------#
def subtract(a,b):
    """ returns the difference between
        a and b """ 
    c = a - b
    c[c == -np.inf] = -MAX
    c[c == np.inf] = MAX
    return(c)
#----------------------------------#
def multiply(a,b):
    """ returns the multiple of a and b """
    c = a * b
    c[c == -np.inf] = -MAX
    c[c == np.inf] = MAX
    return(c)

###################### LOGIC OPERATORS #################################

def not_equal(a,b):
    """ return 1 if True, else 0 """
    c = (a != b).astype(int)
    return(c)
#----------------------------------#
def equal(a,b):
    """ return 1 if True, else 0 """
    c = (a == b).astype(int)
    return(c)
#----------------------------------#
def lt(a,b):
    """ return 1 if True, else 0 """
    c = (a < b).astype(int)
    return(c)
#----------------------------------#
def gt(a,b):
    """ return 1 if True, else 0 """
    c = (a > b).astype(int)
    return(c)
#----------------------------------#
def OR(a,b):
    """ return 1 if True, else 0 """
    c = np.logical_or(a != 0, b != 0)
    return(c)
#----------------------------------#
def xor(a,b):
    """ do xor on values anded with 0 """
    a_not_b = np.logical_and(a != 0, b == 0)
    b_not_a = np.logical_and(a == 0, b != 0)
    c = np.logical_or(a_not_b, b_not_a)
    return(c)
#----------------------------------#
def AND(a,b):
    """ logical and of a and b """
    c = np.logical_and(a != 0, b != 0)
    return(c)
####################### BITWISE OPERATORS ##############################
def bitand(a,b):
    """ bitwise AND: 110 & 101 eq 100 """
    int_a = a.astype(int)
    int_b = b.astype(int)
    c = np.bitwise_and(a, b)
    return(c)
#----------------------------------#
def bitor(a,b):
    """ bitwise OR: 110 | 101 eq 111 """
    int_a = a.astype(int)
    int_b = b.astype(int)
    c = np.bitwise_or(a, b)
    return(c)
#----------------------------------#
def bitxor(a,b):
    """ bitwise XOR: 110 ^ 101 eq 011 """
    int_a = a.astype(int)
    int_b = b.astype(int)
    c = np.bitwise_xor(a, b)
######################## UNARY OPERATORS ###############################
def ABS(a):
    """ return absolute value """
    c = np.abs(a)
    return(c)
#----------------------------------#
def factorial(a):
    """ returns 0 if a >= 100 """
    f = np.abs(np.round(a, 0).astype(int))
    f[f > FACTMAX] = FACTMAX
    f_matrix = np.ones((len(f), np.max(f)))
    f_matrix = f.reshape(-1,1)*f_matrix
    f_matrix -= np.arange(0, np.max(f))
    f_matrix[f_matrix < 1] = 1
    c = np.product(f_matrix, axis = 1)
    return(c)
#----------------------------------#
def NOT(a):
    c = (a == 0).astype(int) 
    return(a)
#----------------------------------#
def log10ofA(a):
    """ Return the logarithm of a to base 10. """
    c = np.log10(a, where = a > 0) 
    c[c < 0] *= -1
    c[c == 0] = 1
    return(c)
#----------------------------------#
def log2ofA(a):
    """ Return the logarithm of a to base 2. """
    c = np.log2(a, where = a > 0) 
    c[c < 0] *= -1
    c[c == 0] = 1
    return(c)
#----------------------------------#
def logEofA(a):
    """ Return the natural logarithm of a. """
    c = np.log(a, where = a > 0) 
    c[c < 0] *= -1
    c[c == 0] = 1
    return(c)

######################## LARGE OPERATORS ###############################
def power(a,b):
    """ return a to the power of b """
    a = np.abs(a)  # ensure the denial of complex number creation
    b[b > 100] = 100
    c = a**b
    c[c > MAX] = MAX 
    return z
#----------------------------------#
def logAofB(a,b):
    """ Return the logarithm of a to the given b. """
    alog = np.log2(a, where = a > 0) 
    alog[alog  < 0] *= -1
    alog[alog== 0] = 1
    blog = np.log2(b, where = b > 0) 
    blog[blog  < 0] *= -1
    blog[blog== 0] = 1
    c = alog/blog
    c[np.abs(c) == np.inf] = 1
    return(c)
#----------------------------------#
# TODO: confirm original author's intent with this one. 
def permute(a,b):
    """ reordering elements """
    a1 = np.abs(round(a))
    b1 = np.abs(round(b))
    indices = b1 > a1
    a2 = b1[indices]
    b2 = a1[indices]
    b1[indices] = b2
    a1[indices] = a2
    c = factorial(a1)/factorial(a1 - b1)
    return(c)
#----------------------------------#
def choose(a,b):
    """ n Choose r function """
    a1 = np.abs(np.round(a, 0))
    b1 = np.abs(np.round(b, 0))
    indices = b1 > a1
    a2 = b1[indices]
    b2 = a1[indices]
    b1[indices] = b2
    a1[indices] = a2
    c =  factorial(a)/(factorial(b)*factorial(a-b))
    return(c)
######################### MISC OPERATORS ###############################
def minimum(a,b):
    """ return the smallest value of a and b """
    c = np.min([a, b], axis = 0)
    return(c)
#----------------------------------#
def maximum(a,b):
    """ return the largest value of a and b """
    c = np.max([a, b], axis = 0)
    return(c)
#----------------------------------#
def left(a, b):
    """ return left value """
    return(a)
#----------------------------------#
def right(a, b):
    """ return right value """
    return(b)
