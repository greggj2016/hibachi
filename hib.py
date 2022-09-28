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
from MI_library import compute_MI
from geno_sim_library import simulate_correlated_SNPs
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
# import re
###############################################################################
if (sys.version_info[0] < 3):
    printf("Python version 3.5 or later is HIGHLY recommended")
    printf("for speed, accuracy and reproducibility.")

# deap location: C:\Users\John Gregg\miniconda3\envs\hibachi\Lib\site-packages\deap

start = time.time()

maf = options['minor_allele_freq']
cov = options['covariance_info']
num_effects = options['num_effects']
python_scoop = options['python_scoop']
infile = options['file']
try:
    evaluate = int(options['evaluation'])
except:
    evaluate = options['evaluation']
population = options['population']
generations = options['generations']
rdf_count = options['random_data_files']
ig_weights = options['information_gain']
lam = options['L2_penalty']
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
    if len(maf) == 1:
        maf = np.repeat(maf, cols)
    if type(cov) == type("file.txt"):
        cov_info = pd.read_csv(cov, delimiter = "\t", header = None)
        cov_info = cov_info.to_numpy().reshape(-1)
        data = simulate_correlated_SNPs(maf, cov_info, rows).astype(np.int8)
    elif type(cov) == type(0.5):
        cov_info = cov*np.ones(np.sum(np.arange(len(maf))))
        data = simulate_correlated_SNPs(maf, cov_info, rows).astype(np.int8)
    else:
        probs = np.array([(1 - maf)**2, 2*maf*(1 - maf), maf**2]).T
        data = np.zeros((rows, cols), dtype = np.int8)
        for i in range(cols):
            data[:, i] += np.random.choice(a = [0, 1, 2], size = rows, p = probs[i])
    x = data.T
else:
    data, x = IO.read_file(infile)
    rows = len(data)
    cols = len(x)

#from copy import deepcopy as COPY
#x_COPY = COPY(x)
#data_COPY = COPY(data)

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
    igvars = np.array([])
    x = xdata
    data = xtranspose
    # Transform the tree expression in a callable function
    names = [el.name for el in individual]
    inds = [int(n[1:]) for n in names if n[0] == 'X' and n[1] in np.arange(10).astype(str)]
    inds = np.unique(inds)
    func = toolbox.compile(expr=individual)

    # Create class possibility.  
    # If class has a unique length of 1, toss it.
    try:
        result = func(*x)
        if (len(np.unique(result)) == 1):
            return -sys.maxsize, sys.maxsize
        result = evals.reclass_result(func(*x), prcnt)

    except:
        print("!!!!!!!!!!!!!ERROR!!!!!!!!!")
        return -sys.maxsize, sys.maxsize
     
    if evaluate in ['normal', 'oddsratio'] or type(evaluate) == type(2):
        rangeval = numfolds = 1  # must be equal
        x_folds = evals.getfolds(x, numfolds)

    elif evaluate == 'folds':
        rangeval = numfolds = 10  # must be equal
        x_folds = evals.getfolds(x, numfolds)

    elif evaluate == 'subsets':
        rangeval = 10
        percent = 25

    elif evaluate == 'noise':
        rangeval = 10
        percent = 10
    
    y_folds = evals.getfolds(np.array(result).reshape(1,-1), numfolds)

    for m in range(rangeval):
        igsum = 0 

        if evaluate == 'subsets': 
            xsub = evals.subsets(x,percent)

        elif evaluate == 'noise': 
            xsub = evals.addnoise(x,percent)

        else:  # normal or folds
            ig_sum_set, ig_var_set = [], []
            ig_vec = np.where(np.array(ig_weights) != None)[0] + 1
            for ig, w in zip(ig_vec, ig_weights):
                if len(inds) < ig and w != 0: 
                    igsum = 0
                    igvar = 1
                    ig_sum_set.append(igsum)
                    ig_var_set.append(igvar)
                elif w != 0: 
                    index_sets = np.array(list(itertools.combinations(inds, ig)))
                    data8, result8 = (x_folds[m].T).astype(np.int8), np.array(y_folds[m]).astype(np.int8)
                    if type(evaluate) == type(2): 
                        indices = np.random.choice(np.arange(len(data8)), evaluate, replace = False)
                        data8, result8 = data8[indices], result8[0, indices]
                    out = [compute_MI(data8[:, i], result8.reshape(-1,1)) for i in index_sets]
                    ig_vals = w*np.array([MI[-1][0] for MI in out])
                    igsum = np.sum(ig_vals)
                    if len(ig_vals) < num_effects: 
                        all_parts = np.zeros(num_effects)
                        all_parts[:len(ig_vals)] += ig_vals
                        igvar = np.var(all_parts)
                    else:                   
                        igvar = np.var(np.sort(ig_vals)[-num_effects:])
                    ig_sum_set.append(igsum)
                    ig_var_set.append(igvar)
            igsums = np.append(igsums, np.sum(ig_sum_set))
            igvars = np.append(igvars, np.sum(ig_var_set))

    if evaluate == 'oddsratio':
        sum_of_diffs, OR = evals.oddsRatio(xsub, result, inst_length)
        individual.OR = OR
        individual.SOD = sum_of_diffs
        individual.igsum = igsum
        
    igsum_avg = np.mean(igsums)        
    igvar_avg = np.mean(igvars)
    fit_fun = igsum_avg - lam*igvar_avg
    #if igsum_avg > 0.5:
    #    pdb.set_trace()
    # print(fit_fun)
    # pdb.set_trace()
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

            return fitfun, len(individual) / float(uniq_col_count), sum_of_diffs
        elif evaluate == 'normal':
            return fit_fun, len(individual)
        else:
            return fit_fun, len(individual)

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
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

    pop, log = algorithms.eaMuPlusLambda(pop,toolbox,mu=MU,lambda_=LAMBDA, 
                          cxpb=0.70, mutpb=0.30, ngen=NGEN, stats=stats, 
                          verbose=True, halloffame=hof)
    return pop, stats, hof, log
##############################################################################
# run the program
##############################################################################

if __name__ == "__main__":
    
    time1 = time.strftime("%H:%M:%S", time.localtime())
    printf("start time:  %s\n", time1)
    printf("input data:  %s\n", infile)
    printf("data shape:  %d X %d\n", rows, cols)
    printf("random seed: %d\n", rseed)
    printf("prcnt cases: %d%%\n", prcnt)
    printf("output dir:  %s\n", outdir)
    if(model_file == ""):
        printf("population:  %d\n", population)
        printf("generations: %d\n", generations)
        printf("evaluation:  %s\n", evaluate)
        printf("ign [1way, 2way, 3way]:  %s\n", str(ig_weights))
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
    best = np.array([ind for ind in hof], dtype = object)
    fitness = np.array([b.fitness.values for b in best])
    penalized_igs, lengths = [f[0] for f in fitness], [f[1] for f in fitness]
    best_index = np.argmax(penalized_igs)
    best_model = best[best_index]
    best_func = toolbox.compile(expr=best_model)
    best_output = best_func(*x).astype(np.float32)
    best_labels = evals.reclass_result(best_output, prcnt)

    '''
    names = [el.name for el in best_model]
    inds = [int(n[1:]) for n in names if n[0] == 'X' and n[1] in np.arange(10).astype(str)]
    inds = np.unique(inds) - 1
    index_sets = np.array(list(itertools.combinations(inds, ig)))
    out = [compute_MI(data[:, i], best_labels.reshape(-1,1)) for i in index_sets]
    ig_vals = np.array([MI[-1][0] for MI in out])
    igsum = np.sum(ig_vals)
    igvar = np.var(np.sort(ig_vals)[-ig:])
    '''

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

    time2 = time.strftime("%H:%M:%S", time.localtime())
    printf("computation end time:  %s\n", time2)

    tottime = time.time() - start
    if tottime > 3600:
        printf("\nRuntime: %.2f hours\n", tottime/3600)
    elif tottime > 60:
        printf("\nRuntime: %.2f minutes\n", tottime/60)
    else:
        printf("\nRuntime: %.2f seconds\n", tottime)

    time3 = time.strftime("%H:%M:%S", time.localtime())
    printf("time statements end time:  %s\n", time3)

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
    ig_label = str(ig_weights[0]) + 'ig1_' + str(ig_weights[1])  + 'ig2_' + str(ig_weights[2]) + 'ig3'
    outfile += str(evaluate) + "-" + ig_label
    outfile += "_lam" + str(lam) + "_effs" + str(num_effects) + ".txt" 

    time4 = time.strftime("%H:%M:%S", time.localtime())
    printf("file definition end time:  %s\n", time4)

    printf("writing data with Class to %s\n", outfile)

    IO.create_file(x, best_labels, outfile)       # use first individual
    #
    # write top model out to file
    #
    time6 = time.strftime("%H:%M:%S", time.localtime())
    printf("file writing end time:  %s\n", time6)

    moutfile = outdir + "model-" + file1 + "-" + rowxcol + '-' 
    moutfile += 's' + str(rseed) + '-' 
    moutfile += popstr + '-'
    moutfile += genstr + '-'
    ig_label = str(ig_weights[0]) + 'ig1_' + str(ig_weights[1])  + 'ig2_' + str(ig_weights[2]) + 'ig3'
    moutfile += str(evaluate) + "-" + ig_label
    moutfile += "_lam" + str(lam) + "_effs" + str(num_effects) + ".txt" 
    printf("writing model to %s\n", moutfile)
    time7 = time.strftime("%H:%M:%S", time.localtime())
    printf("file2 definition end time:  %s\n", time7)

    IO.write_model(moutfile, best)
    time8 = time.strftime("%H:%M:%S", time.localtime())
    printf("file2 writing end time:  %s\n", time8)

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
            outfile += str(evaluate) + '-' + str(ig) + "way"
            outfile += "_lam" + str(lam) + "_effs" + str(num_effects) + ".txt" 
            printf("%s\n", outfile)
            IO.create_file(X,nresult,outfile)

    time9 = time.strftime("%H:%M:%S", time.localtime())
    printf("other1 end time:  %s\n", time9)

    #
    # plot data if selected
    #
    file = os.path.splitext(os.path.basename(infile))[0]
    if Stats: # (-S)
        statfile = outdir + "stats-" + file + "-" + str(evaluate) 
        statfile += "-" + str(rseed) + ".pdf"
        printf("saving stats to %s\n", statfile)
        plots.plot_stats(df,statfile)

    time10 = time.strftime("%H:%M:%S", time.localtime())
    printf("other2 end time:  %s\n", time10)

    if Trees: # (-T)
        treefile = outdir + 'tree_' + str(save_seed) + '.pdf'
        printf("saving tree plot to %s\n", treefile)
        plots.plot_tree(best[0],save_seed,outdir)

    time11 = time.strftime("%H:%M:%S", time.localtime())
    printf("other3 end time:  %s\n", time11)

    if Fitness == True: # (-F)
        outfile = outdir
        outfile += "fitness-" + file + "-" + str(evaluate) + "-" + str(rseed) + ".pdf"
        printf("saving fitness plot to %s\n", outfile)
        plots.plot_fitness(fitness,outfile)

    time12 = time.strftime("%H:%M:%S", time.localtime())
    printf("other4 end time:  %s\n", time12)