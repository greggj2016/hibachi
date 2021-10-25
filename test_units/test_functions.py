from inspect import getmembers
from inspect import isfunction
from inspect import signature
from itertools import product
from copy import deepcopy as COPY
from pandas import read_csv
from pandas import DataFrame as df
from itertools import combinations as combos
import numpy as np
import os
import sys
import math
import unittest
import pdb

import operators_old
from utils import three_way_information_gain as three_way_ig_old
from utils import two_way_information_gain as two_way_ig_old
from utils import mutual_information as one_way_ig_old

# choose is bugged in the original because math.factorial can't handle
# arbitrarily large numbers, but it still runs and returns incorrect results. 
# permute is bugged in the original for the same reason as choose
# gt is functionally equivalent to "less than or equal to" in the original
# These original functions are therefore excluded from the
# comparison to my modified versions of the same functions.
np.random.seed(0)
FACTMAX_old = 170
largest = math.factorial(FACTMAX_old)
FACTMAX_new = 20.
MAX = np.product(FACTMAX_new  - np.arange(0, FACTMAX_new))

#path finding code start
next_dir, next_folder = os.path.split(os.getcwd())
main_folder = "hibachi"
count = 1
paths = [os.getcwd(), next_dir]
folders = ["", next_folder]
while next_folder != main_folder and count < 4:
    next_dir, next_folder = os.path.split(next_dir)
    paths.append(next_dir)
    folders.append(next_folder)
    count += 1
if count >= 4:
    message = "error: important paths have been renamed or reorganized. "
    message += "If this was intentional, then change the path "
    message += "finding code in test_main_library.py"
    print(message)
os.chdir(paths[count - 1])
sys.path.insert(1, os.getcwd())
#path finding code end

import operators
from MI_library import compute_MI as n_way_ig_new

def remove_infinities(vectors):
    for i in range(len(vectors)):
        vectors[i][vectors[i] == np.inf] = MAX
        vectors[i][vectors[i] == -np.inf] = -MAX
    return(vectors)

def round_mag(arr):
    arr2 = np.abs(np.array(arr).astype(float))
    arr2[arr2 < 1] = 1
    arr2 = np.floor(np.log10(arr2))
    return(10**arr2)

functions = df(getmembers(operators, isfunction))
functions_old = df(getmembers(operators_old, isfunction))
all_functions = functions.merge(functions_old, how = "inner", on = 0).to_numpy()

numbers1 = np.random.normal(0, 1, 1000)
numbers2 = np.random.normal(0, 1, 1000)
binary1 = np.round(np.random.rand(1000), 0)
binary2 = np.round(np.random.rand(1000), 0)
pzeros = np.zeros(1000)
nzeros = -np.zeros(1000)
pinf = np.inf*np.ones(1000)
ninf = -np.inf*np.ones(1000)
cases = [numbers1, numbers2, binary1, binary2, pzeros, nzeros, pinf, ninf]
case_names = ["num1", "num2", "bin1", "bin2", "+0", "-0", "+inf", "-inf"]
case_pairs = list(product(case_names, repeat = 2))
    
class test_functions(unittest.TestCase):

    def test_primitive_correctness(self):

        new_output = []
        old_output = []
        f_names = []
        for f_name, f_new, f_old in all_functions:
            nargs = len(signature(f_new).parameters)
            for vectors in product(cases, repeat = nargs):
                new_output.append(f_new(*vectors))
                vectors = remove_infinities(COPY(vectors))
                result = []
                for row in np.array(vectors).T: result.append(f_old(*row))
                result = [max(min(MAX, i), -MAX) for i in result]
                f_names.append(f_name)
                old_output.append(np.array(result))

        new_output2 = np.round(np.array(new_output).astype(float)/round_mag(new_output), 8)
        old_output2 = np.round(np.array(old_output).astype(float)/round_mag(old_output), 8)
        indices = np.all(new_output2 == old_output2, axis = 1) == False
        different_functions = np.array(f_names)[indices]
        right_answer = np.array(16*['choose'] + 64*['gt'] +  16*['permute'], dtype='<U12')
        message = "At least 1 hibachi function is inconsistent with the previous version"
        self.assertTrue(np.array_equal(different_functions, right_answer), message)

    def test_primitive_units(self):

        path = "test_units/expected_primitive_output.txt"
        expected_output = read_csv(path, delimiter = "\t", header = None).to_numpy(float)
        expected_output = np.round(expected_output/round_mag(expected_output), 8)
        equivalency = []
        i = 0
        for f_name, f_new, f_old in all_functions:
            nargs = len(signature(f_new).parameters)
            for vectors in product(cases, repeat = nargs):
                expected = expected_output[i]
                i += 1
                actual = np.round(f_new(*vectors).astype(float)/round_mag(f_new(*vectors)), 8)
                equivalency.append(np.all(expected == actual))

        message = "at least one primitive failed to produce the expected output"
        self.assertTrue(np.all(equivalency), message)

    def test_MI_computation_units(self):

        N = 50000
        X = (np.random.rand(3, N)*(3 - 1E-10)).astype(int)
        y = X[0] + X[1] < X[2]
        old_ig3 = np.array([three_way_ig_old(X[0], X[1], X[2], y)])
        old_ig2 = np.array([two_way_ig_old(X[i], X[j], y) for i, j in combos([0, 1, 2], 2)])
        old_ig1 = np.array([one_way_ig_old(X[i], y) for i in [0, 1, 2]])
        old_ig = [old_ig1, old_ig2, old_ig3]
        old_ig = [np.round(old, 7) for old in old_ig]
        new_ig = n_way_ig_new(X.T, y.reshape(-1, 1))
        new_ig = [np.round(new, 7) for new in new_ig]
        equivalency = np.all([np.all(old == new) for old, new in zip(old_ig, new_ig)])
        message = "functions in MI_library do not produce consistent output with utils MI functions"
        self.assertTrue(equivalency, message)
        
if __name__ == '__main__':
    unittest.main()
