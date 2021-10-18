#!/usr/bin/env python
#===============================================================================
#
#          FILE:  hib.py
# 
#         USAGE:  ./hib.py [options]
# 
#   DESCRIPTION:  Data simulation software that creates data sets with 
#                 particular characteristics
#
#       OPTIONS:  ./hib.py -h for all options
#
#  REQUIREMENTS:  python >= 3.5, deap, scikit-mdr, pygraphviz
#          BUGS:  Damn ticks!!
#       UPDATES:  170224: try/except in evalData()
#                 170228: files.sort() to order files
#                 170313: modified to use IO.get_arguments()
#                 170319: modified to use evals for evaluations
#                 170320: modified to add 1 to data elements before processing
#                 170323: added options for plotting
#                 170410: added call to evals.reclass_result() in evalData()
#                 170417: reworked post processing of new random data tests
#                 170422: added ability for output directory selection
#                         directory is created if it doesn't exist
#                 170510: using more protected operators from operators.py
#                 170621: import information gains from local util.py to
#                         avoid unnecessary matplotlib import
#                 170626: added equal and not_equal operators
#                 170706: added option to show all fitnesses
#                 170710: added option to process given model
#                         writes out best model to model file
#                 180307: added oddsratio as an option to evaluate
#                 180316: added tree plot (-T) to model file processing (-m)
#       AUTHORS:  Pete Schmitt (discovery), pschmitt@upenn.edu
#                 Randy Olson, olsonran@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.3.1
#       CREATED:  02/06/2017 14:54:24 EST
#      REVISION:  Fri Mar 16 14:54:07 EDT 2018
#===============================================================================
import IO
options = IO.get_arguments()
from IO import printf
from deap import algorithms, base, creator, tools, gp
from utils import three_way_information_gain as three_way_ig
from MI_library import compute_MI
from utils import two_way_information_gain as two_way_ig
from joblib import Parallel
from joblib import delayed
import evals
import itertools
import glob
import numpy as np
import operator as op
import operators as ops
import os
import pandas as pd
import random
import sys
import time
import pdb
import re
###############################################################################
if (sys.version_info[0] < 3):
    printf("Python version 3.5 or later is HIGHLY recommended")
    printf("for speed, accuracy and reproducibility.")

# deap location: C:\Users\John Gregg\miniconda3\envs\hibachi\Lib\site-packages\deap

labels = []
all_igsums = []
#results = []
start = time.time()

python_scoop = options['python_scoop']
infile = options['file']
evaluate = options['evaluation']
population = options['population']
generations = options['generations']
rdf_count = options['random_data_files']
ig = options['information_gain']
rows = options['rows']
njobs = options['njobs']
cols = options['columns']
Stats = options['statistics']
Trees = options['trees']
Fitness = options['fitness']
prcnt = options['percent']
outdir = options['outdir']
showall = options['showallfitnesses']
model_file = options['model_file']
#
# define output variables
#
rowxcol = str(rows) + 'x' + str(cols)
popstr = 'p' + str(population)
genstr = 'g' + str(generations)
#
if Fitness or Trees or Stats:
    import plots
#
# set up random seed
#
elif(options['seed'] == -999):
    rseed = random.randint(1,1000)
else:
    rseed = options['seed']
random.seed(rseed)
np.random.seed(rseed)
#
# Read/create the data and put it in a list of lists.
# data is normal view of columns as features
# x is transposed view of data
#
if infile == 'random':
    data = (2.99999999*np.random.rand(rows, cols)).astype(int)
    x = data.T
else:
    data, x = IO.read_file(infile)
    rows = len(data)
    cols = len(x)

inst_length = len(x)
###############################################################################

# defined a new primitive set for strongly typed GP
arr = np.ndarray
in_types = itertools.repeat(arr, inst_length)
pset = gp.PrimitiveSetTyped("MAIN", in_types, arr, "X")

# basic operators
pset.addPrimitive(ops.addition, [arr, arr], arr)
pset.addPrimitive(ops.subtract, [arr, arr], arr)
pset.addPrimitive(ops.multiply, [arr, arr], arr)
pset.addPrimitive(ops.safediv, [arr, arr], arr)
pset.addPrimitive(ops.modulus, [arr, arr], arr)
pset.addPrimitive(ops.plus_mod_two, [arr, arr], arr)

# logic operators 
pset.addPrimitive(ops.equal, [arr, arr], arr)
pset.addPrimitive(ops.not_equal, [arr, arr], arr)
pset.addPrimitive(ops.gt, [arr, arr], arr)
pset.addPrimitive(ops.lt, [arr, arr], arr)
pset.addPrimitive(ops.AND, [arr, arr], arr)
pset.addPrimitive(ops.OR, [arr, arr], arr)
pset.addPrimitive(ops.xor, [arr, arr], arr)

# bitwise operators 
pset.addPrimitive(ops.bitand,[arr, arr], arr)
pset.addPrimitive(ops.bitxor, [arr, arr], arr)

# unary operators 
pset.addPrimitive(ops.ABS, [arr], arr)
pset.addPrimitive(ops.NOT, [arr], arr)
pset.addPrimitive(ops.factorial, [arr], arr)
pset.addPrimitive(ops.left, [arr, arr], arr)
pset.addPrimitive(ops.right, [arr, arr], arr)

# large operators 
pset.addPrimitive(ops.power, [arr, arr], arr)
pset.addPrimitive(ops.logAofB, [arr, arr], arr)
pset.addPrimitive(ops.permute, [arr, arr], arr)
pset.addPrimitive(ops.choose, [arr, arr], arr)

# misc operators 
pset.addPrimitive(ops.minimum, [arr, arr], arr)
pset.addPrimitive(ops.maximum, [arr, arr], arr)

# terminals 
timeval = str(int(time.time()*1E16))
# so it can rerun from ipython and concurrent processes sharing a random seed. 
pset.addEphemeralConstant(timeval, lambda: random.random() * 100, float)
pset.addTerminal(0.0, float)
pset.addTerminal(1.0, float)

# creator: the parent process becomes a child process, and when it does,
# it reruns all of its code. This overwrites its own FitnessMulti and Individual
# Classes, which is not problematic, but it does produce a warning. The try
# statement below ensures that the functions will not be overwritten.
try:
    globals()[str(os.getpid())]
except:
    globals()[str(os.getpid())] = 1
    if evaluate == 'oddsratio':
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
    else:
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

# toolbox 
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
toolbox.register("individual",
                 tools.initIterate,creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

##############################################################################
def evalData(individual, xdata, xtranspose):

    # """ access the individual's functional complexity if needed"""
    # elements =  np.array(re.split(' |\(|\)|,', str(individual)))
    # elements = elements[elements != '']
    # is_var = np.array([el[0] == 'X'for el in elements])
    # variables = elements[is_var]
    # functions = elements[is_var == False]
    
    """ evaluate the individual """
    result = []
    igsums = np.array([])
    x = xdata
    data = xtranspose
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
        
    # Create class possibility.  
    # If class has a unique length of 1, toss it.
    try:
        result = func(*x)
    except:
        return -sys.maxsize, sys.maxsize

    if (len(np.unique(result)) == 1):
        return -sys.maxsize, sys.maxsize
     
    if evaluate == 'normal' or evaluate == 'oddsratio':
        rangeval = 1

    elif evaluate == 'folds':
        rangeval = numfolds = 10  # must be equal
        folds = evals.getfolds(x, numfolds)

    elif evaluate == 'subsets':
        rangeval = 10
        percent = 25

    elif evaluate == 'noise':
        rangeval = 10
        percent = 10

    result = evals.reclass_result(x, result, prcnt)

    for m in range(rangeval):
        igsum = 0 
        if evaluate == 'folds': 
            xsub = list(folds[m])

        elif evaluate == 'subsets': 
            xsub = evals.subsets(x,percent)

        elif evaluate == 'noise': 
            xsub = evals.addnoise(x,percent)

        else:  # normal
            xsub = x

        # Calculate information gain between data columns and result
        # and return mean of these calculations
        if(ig == 2): 
            index_sets = np.array(list(itertools.combinations(range(inst_length), 2)))
            out = Parallel(n_jobs = njobs)(delayed(compute_MI)(data[:, i], np.array(result).reshape(-1,1)) for i in index_sets)
            igsum = np.sum([MI[-1][0] for MI in out])
        elif(ig == 3):
            index_sets = np.array(list(itertools.combinations(range(inst_length), 3)))
            out = Parallel(n_jobs = njobs)(delayed(compute_MI)(data[:, i], np.array(result).reshape(-1,1)) for i in index_sets)
            igsum = np.sum([MI[-1][0] for MI in out])

        igsums = np.append(igsums,igsum)
        
    if evaluate == 'oddsratio':
        sum_of_diffs, OR = evals.oddsRatio(xsub, result, inst_length)
        individual.OR = OR
        individual.SOD = sum_of_diffs
        individual.igsum = igsum
        
    igsum_avg = np.mean(igsums)
    labels.append((igsum_avg, result)) # save all results
    all_igsums.append(igsums)

    if len(individual) <= 1:
        return -sys.maxsize, sys.maxsize
    else:
        if evaluate == 'oddsratio':
            individual_str = str(individual)
            uniq_col_count = 0
            for col_num in range(len(x)):
                col_name = 'X{}'.format(col_num)
                if col_name in individual_str:
                    uniq_col_count += 1

            return igsum, len(individual) / float(uniq_col_count), sum_of_diffs
#           return igsum, len(individual), sum_of_diffs
        elif evaluate == 'normal':
            return igsum, len(individual)
        else:
            return igsum_avg, len(individual)

##############################################################################
#def parmap(function, list_input):
#    return(Parallel(n_jobs = njobs)(delayed(function)(i) for i in list_input))
#toolbox.register("map", parmap)
if python_scoop == True:
    from scoop import futures
    toolbox.register("map", futures.map)
toolbox.register("evaluate", evalData, xdata = x, xtranspose=data)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
##############################################################################
def pareto_eq(ind1, ind2):
    """Determines whether two individuals are equal on the Pareto front
       Parameters (ripped from tpot's base.py)
        ----------
        ind1: DEAP individual from the GP population
         First individual to compare
        ind2: DEAP individual from the GP population
         Second individual to compare
        Returns
        ----------
        individuals_equal: bool
         Boolean indicating whether the two individuals are equal on
         the Pareto front
    """
    return np.all(ind1.fitness.values == ind2.fitness.values)
##############################################################################
def hibachi(pop,gen,rseed,showall):
    """ set up stats and population size,
        then start the process """
    MU, LAMBDA = pop, pop
    NGEN = gen 
    np.random.seed(rseed)
    random.seed(rseed)
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront(similar=pareto_eq)
    if showall:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
    else:
        stats = tools.Statistics(lambda ind: max(ind.fitness.values[0],0))
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

    pop, log = algorithms.eaMuPlusLambda(pop,toolbox,mu=MU,lambda_=LAMBDA, 
                          cxpb=0.7, mutpb=0.3, ngen=NGEN, stats=stats, 
                          verbose=True, halloffame=hof)
    return pop, stats, hof, log
##############################################################################
# run the program
##############################################################################

if __name__ == "__main__":
    
    printf("input data:  %s\n", infile)
    printf("data shape:  %d X %d\n", rows, cols)
    printf("random seed: %d\n", rseed)
    printf("prcnt cases: %d%%\n", prcnt)
    printf("output dir:  %s\n", outdir)
    if(model_file == ""):
        printf("population:  %d\n", population)
        printf("generations: %d\n", generations)
        printf("evaluation:  %s\n", evaluate)
        printf("ign 2/3way:  %d\n", ig)
    printf("\n")
    # 
    # If model file, ONLY process the model
    #
    if(model_file != ""):
        individual = IO.read_model(model_file)
        func = toolbox.compile(expr=individual)
        result = [(func(*inst[:inst_length])) for inst in data]
        nresult = evals.reclass_result(x, result, prcnt)
        outfile = outdir + 'results_using_model_from_' + os.path.basename(model_file) 
        printf("Write result to %s\n", outfile)
        IO.create_file(x,nresult,outfile)
        if Trees == True: # (-T)
            M = gp.PrimitiveTree.from_string(individual,pset)
            outtree = outdir + 'tree_' + str(rseed) + '.pdf'
            printf("saving tree plot to %s\n", outtree)
            plots.plot_tree(M,rseed,outdir)
        sys.exit(0)
    #
    # Start evaluation here
    #
    pop, stats, hof, logbook = hibachi(population,generations,rseed,showall)
    best = []
    fitness = []
    for ind in hof:
        best.append(ind)
        fitness.append(ind.fitness.values)

    printf("\n")
    printf("IND\tFITNESS\t\tMODEL\n")
    for i in range(len(hof)):
        printf("%d\t%.8f\t%s\n", i, fitness[i][0], str(best[i]))

    if evaluate == 'oddsratio':
        IO.create_OR_table(best,fitness,rseed,outdir,rowxcol,popstr,
                           genstr,evaluate,ig)

    record = stats.compile(pop)
    printf("statistics: \n")
    printf("%s\n", str(record))

    tottime = time.time() - start
    if tottime > 3600:
        printf("\nRuntime: %.2f hours\n", tottime/3600)
    elif tottime > 60:
        printf("\nRuntime: %.2f minutes\n", tottime/60)
    else:
        printf("\nRuntime: %.2f seconds\n", tottime)
    df = pd.DataFrame(logbook)
    del df['gen']
    del df['nevals']
    #
    # sys.exit(0)
    #
    if(infile == 'random'):
        file1 = 'random0'
    else:
        file1 = os.path.splitext(os.path.basename(infile))[0]
    #
    # make output directory if it doesn't exist
    #
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outfile = outdir + "results-" + file1 + "-" + rowxcol + '-' 
    outfile += 's' + str(rseed) + '-' 
    outfile += popstr + '-'
    outfile += genstr + '-'
    outfile += evaluate + "-" + 'ig' + str(ig) + "way.txt" 
    printf("writing data with Class to %s\n", outfile)
    labels.sort(key=op.itemgetter(0),reverse=True)     # sort by igsum (score)
    IO.create_file(x,labels[0][1],outfile)       # use first individual
    #
    # write top model out to file
    #
    moutfile = outdir + "model-" + file1 + "-" + rowxcol + '-' 
    moutfile += 's' + str(rseed) + '-' 
    moutfile += popstr + '-'
    moutfile += genstr + '-'
    moutfile += evaluate + "-" + 'ig' + str(ig) + "way.txt" 
    printf("writing model to %s\n", moutfile)
    IO.write_model(moutfile, best)
    #
    #  Test remaining data files with best individual (-r)
    #
    save_seed = rseed
    if(infile == 'random' and rdf_count > 0):
        printf("number of random data to generate: %d\n", rdf_count)
        for i in range(rdf_count):
            rseed += 1
            D, X = IO.get_random_data(rows,cols,rseed)
            nfile = 'random' + str(i+1)
            printf("%s\n", nfile)
            individual = best[0]
            func = toolbox.compile(expr=individual)
            result = [(func(*inst[:inst_length])) for inst in D]
            nresult = evals.reclass_result(X, result, prcnt)
            outfile = outdir + 'model_from-' + file1 
            outfile += '-using-' + nfile + '-' + str(rseed) + '-' 
            outfile += str(evaluate) + '-' + str(ig) + "way.txt" 
            printf("%s\n", outfile)
            IO.create_file(X,nresult,outfile)
    #
    # plot data if selected
    #
    file = os.path.splitext(os.path.basename(infile))[0]
    if Stats: # (-S)
        statfile = outdir + "stats-" + file + "-" + evaluate 
        statfile += "-" + str(rseed) + ".pdf"
        printf("saving stats to %s\n", statfile)
        plots.plot_stats(df,statfile)

    if Trees: # (-T)
        treefile = outdir + 'tree_' + str(save_seed) + '.pdf'
        printf("saving tree plot to %s\n", treefile)
        plots.plot_tree(best[0],save_seed,outdir)

    if Fitness == True: # (-F)
        outfile = outdir
        outfile += "fitness-" + file + "-" + evaluate + "-" + str(rseed) + ".pdf"
        printf("saving fitness plot to %s\n", outfile)
        plots.plot_fitness(fitness,outfile)
