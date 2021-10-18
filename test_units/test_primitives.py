from inspect import getmembers
from inspect import isfunction
from inspect import signature
from itertools import product
import numpy as np
import itertools
import operators
import pdb

functions = getmembers(operators, isfunction)
numbers = np.random.normal(0, 10, 10000)
binary = np.round(np.random.rand(10000), 0)
pzeros = np.zeros(10000)
nzeros = -np.zeros(10000)
pinf = np.inf*np.ones(10000)
ninf = -np.inf*np.ones(10000)
cases = [numbers, pzeros, nzeros, pinf, ninf]

def List(x): return(list(x))
for fname, f in functions:
    if fname in ['COPY']:
        continue
    nargs = len(signature(f).parameters)
    for vectors in product(cases, repeat = nargs):
        f(*vectors)