#!/usr/bin/env python
#==============================================================================
#
#          FILE:  IO.py
# 
#         USAGE:  import IO (from hib.py)
# 
#   DESCRIPTION:  graphing and file i/o routines.  
# 
#       UPDATES:  170213: added subset() function
#                 170214: added getfolds() function
#                 170215: added record shuffle to getfolds() function
#                 170216: added addnoise() function
#                 170217: modified create_file() to name file uniquely
#                 170302: added plot_hist() to plot std
#                 170313: added get_arguments()
#                 170319: added addone()
#                 170329: added np.random.shuffle() to read_file_np() 
#                 170410: added option for case percentage
#                 170420: added option for output directory
#                 170706: added option for showing all fitnesses
#                 170710: added option to process given model
#                         added read_model and write_model
#                 180307: added oddsratio to evaluate options
#                 180514: removed shuffle from read_data()
#        AUTHOR:  Pete Schmitt (discovery (iMac)), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.14
#       CREATED:  02/06/2017 14:54:24 EST
#      REVISION:  Mon May 14 11:50:42 EDT 2018
#==============================================================================
import pandas as pd
import csv
import numpy as np
import argparse
import sys
import os
###############################################################################
def printf(format, *args):
    """ works just like the C/C++ printf function """
    import sys
    sys.stdout.write(format % args)
    sys.stdout.flush()

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
###############################################################################

def get_arguments():
    options = dict()

    parser = argparse.ArgumentParser(description = \
        "Run hibachi evaluations on your data")
    parser.add_argument('-a', '--minor_allele_freq', type=float, nargs = '*',
            help='probability of observing minor allele')
    parser.add_argument('-c', '--covariance_info', type=str,
            help='either a single column file containing one (positive) correlation coefficient for each SNP pairs or a single (positive) value for all pairs')
    parser.add_argument('-di', '--debug_in', type=str,
            help='specifies a input debug log to compare this run against (will dramatically increase runtime)')
    parser.add_argument('-do', '--debug_out', type=str,
            help='specifies a output filename for a debug log (will dramatically increase runtime)')
    parser.add_argument('-e', '--evaluation', type=str,
            help='name of evaluation [normal|folds|subsets|noise|oddsratio]' +
                 ' (default=normal) note: oddsratio sets columns == 10')
    parser.add_argument('-effs', '--num_effects', type=int, help = 'number of effects that you want to simulate')
    parser.add_argument('-f', '--file', type=str,
            help='name of training data file (REQ)' +
                 ' filename of random will create all data')
    parser.add_argument("-g", "--generations", type=int, 
            help="number of generations (default=40)")
    parser.add_argument("-i", "--information_gain", type=int, 
            help="information gain 2 way or 3 way (default=2)")
    parser.add_argument("-m", "--model_file", type=str, 
            help="model file to use to create Class from; otherwise \
                  analyze data for new model.  Other options available \
                  when using -m: [f,o,s,P]")
    parser.add_argument('-n', '--njobs', type=int,
            help='number of parallel jobs for IG calculations (default=1)')
    parser.add_argument('-o', '--outdir', type=str,
            help='name of output directory (default = .)' +
            ' Note: the directory will be created if it does not exist')
    parser.add_argument("-p", "--population", type=int, 
            help="size of population (default=100)")
    parser.add_argument('-ps', '--python_scoop', type=str,
            help='whether or not to use scoop (default=1)')
    parser.add_argument("-r", "--random_data_files", type=int, 
            help="number of random data to use instead of files (default=0)")
    parser.add_argument("-s", "--seed", type=int, 
            help="random seed to use (default=random value 1-1000)")
    parser.add_argument("-y", "--L2_penalty", type=float, 
            help="increasing this makes effects more diverse")
    parser.add_argument("-A", "--showallfitnesses", 
            help="show all fitnesses in a multi objective optimization",
            action='store_true')
    parser.add_argument("-C", "--columns", type=int, 
            help="random data columns (default=3) note: " +
                 "evaluation of oddsratio sets columns to 10")
    parser.add_argument("-F", "--fitness", 
            help="plot fitness results", action='store_true')
    parser.add_argument("-P", "--percent", type=int,
            help="percentage of case for case/control (default=25)")
    parser.add_argument("-R", "--rows", type=int, 
            help="random data rows (default=1000)")
    parser.add_argument("-S", "--statistics", 
            help="plot statistics",action='store_true')
    parser.add_argument("-T", "--trees", 
            help="plot best individual trees",action='store_true')

    args = parser.parse_args()

    if(args.minor_allele_freq == None):
        options['minor_allele_freq'] = np.array([0.5])
    elif(len(args.minor_allele_freq) not in [1, args.columns]):
        message = "\nexiting: user must input either one minor allele "
        message += "frequency for all columns, or they must input one "
        message += "minor allele frequency for each column (i.e. "
        message += "the number OF ARGUMENTS following -a must equal "
        message += "either 1 or the NUMBER following -C)."
        print(message)
        exit()
    elif(np.any(np.array(args.minor_allele_freq) > 0.5)):
        print("\nexiting: all minor allele frequencies must be lower than 0.5")
        exit()
    elif(np.any(np.array(args.minor_allele_freq) <= 0)):
        print("\nexiting: all minor allele frequencies must be higher than 0")
        exit()
    else:
        options['minor_allele_freq'] = np.array(args.minor_allele_freq)

    if(args.covariance_info == None):
        options['covariance_info'] = args.covariance_info
    elif(is_number(args.covariance_info)):
        options['covariance_info'] = float(args.covariance_info)
    else:
        options['covariance_info'] = args.covariance_info

    if(args.debug_out != None and args.debug_in != None):
        print("exiting: user not meant to use both debugging modes simultaneously.")
        print("first use --debug_out FILENAME with an old hibachi version to produce a log of expected output.")
        print("then use --debug_in FILENAME with a new hibachi version to compare its output to the old output in FILENAME.")
        print("the python debugger will run pdb.set_trace() at each detected difference.")
    else:
        options['debug_out'] = args.debug_out
        options['debug_in'] = args.debug_in

    if(args.python_scoop == "True" or args.python_scoop == "true"):
        if(args.debug_out != None or args.debug_in != None):
            print("exiting: multiprocessing and the hibachi debugger may not simultaneously be used")
        options['python_scoop'] = True
    elif(args.python_scoop == "False" or args.python_scoop == "false" or args.python_scoop == None):
        options['python_scoop'] = False
    else:
        print("exiting: unrecognized value for -python_scoop argument")
        exit()

    if(args.num_effects == None):
        options['num_effects'] = 1
    else:
        options['num_effects'] = args.num_effects

    if(args.njobs == None):
        options['njobs'] = 1
    else:
        options['njobs'] = args.njobs

    if(args.file == None):
        printf("filename required\n")
        sys.exit()
    else:
        options['file'] = args.file
        options['basename'] = os.path.basename(args.file)
        options['dir_path'] = os.path.dirname(args.file)

    if(args.model_file != None):
        options['model_file'] = args.model_file
    else:
        options['model_file'] = ""

    if(args.outdir == None):
        options['outdir'] = "./"
    else:
        options['outdir'] = args.outdir + '/'

    if(args.seed == None):
        options['seed'] = -999
    else:
        options['seed'] = args.seed

    if(args.L2_penalty == None):
        options['L2_penalty'] = 0
    elif(args.L2_penalty < 0):
        message = "exiting: L2 penalty must exceed or equal 0."
        print(message)
        exit()
    else:
        options['L2_penalty'] = args.L2_penalty 

    if(args.percent == None):
        options['percent'] = 25
    else:
        options['percent'] = args.percent
        
    if(args.population == None):
        options['population'] = 100
    else:
        options['population'] = args.population

    if(args.information_gain == None):
        options['information_gain'] = 2
    else:
        options['information_gain'] = args.information_gain

    if(args.random_data_files == None):
        options['random_data_files'] = 0
    else:
        options['random_data_files'] = args.random_data_files

    if(args.generations == None):
        options['generations'] = 40
    elif(((args.debug_out) != None or (args.debug_in != None)) and args.generations != 0):
        import pdb
        pdb.set_trace()
        print("generations being set to 0 for debug mode!")
        options['generations'] = 0
    else:
        options['generations'] = args.generations

    if(args.evaluation == None):
        options['evaluation'] = 'normal'
    else:
        options['evaluation'] = args.evaluation
        if options['evaluation'] == 'oddsratio':
            args.columns = 10

    if(args.rows == None):
        options['rows'] = 1000
    else:
        options['rows'] = args.rows

    if(args.columns == None):
        options['columns'] = 3
    else:
        options['columns'] = args.columns

    if(args.showallfitnesses):
        options['showallfitnesses'] = True
    else:
        options['showallfitnesses'] = False

    if(args.statistics):
        options['statistics'] = True
    else:
        options['statistics'] = False

    if(args.trees):
        options['trees'] = True
    else:
        options['trees'] = False

    if(args.fitness):
        options['fitness'] = True
    else:
        options['fitness'] = False

    return options
###############################################################################
def get_random_data(rows, cols, seed=None):
    """ return randomly generated data is shape passed in """
    if seed != None: np.random.seed(seed)
    data = np.random.randint(0,3,size=(rows,cols))
    x = data.transpose()
    return data.tolist(), x.tolist()
###############################################################################
def create_file(x,result,outfile):
    d = np.array(x).transpose()    
    columns = [0]*len(x)
    # create columns names for variable number of columns.
    for i in range(len(x)):
        columns[i] = 'X' + str(i)
    
    df = pd.DataFrame(d, columns=columns)
    
    df['Class'] = result
    df.to_csv(outfile, sep='\t', index=False)
###############################################################################
def read_file(fname):
    """ return both data and x
        data = rows of instances
        x is data transposed to rows of features """
    data = np.genfromtxt(fname, dtype=np.int, delimiter='\t') 
    #np.random.shuffle(data) # give the data a good row shuffle
    x = data.transpose()
    return data.tolist(), x.tolist()
###############################################################################
def write_model(outfile, best):
    """ write top individual out to model file """
    functions = [str(b) for b in best]
    penalized_igs = [b.fitness.values[0] for b in best]
    lengths = [b.fitness.values[1] for b in best]
    model_df = pd.DataFrame(np.array([penalized_igs, lengths, functions]).T)
    model_df.columns = ["penalized_ig", "length", "model"]
    model_df.to_csv(outfile, sep = "\t", header = True, index = False)
###############################################################################
def read_model(infile):
    f = open(infile, 'r')
    m = f.read()
    m = m.rstrip()
    f.close()
    return m
###############################################################################
def create_OR_table(best,fitness,seed,outdir,rowxcol,popstr,
                    genstr,evaluate,ig):
    """ write out odd_ratio and supporting data """
    fname = outdir + "or_sod_igsum-" + rowxcol + '-' 
    fname += 's' + str(seed).zfill(3) + '-'
    fname += popstr + '-' 
    fname += genstr + '-' 
    fname += evaluate + '-ig' + str(ig) + 'way.txt'
    f = open(fname, 'w')
    f.write("Individual\tFitness\tSOD\tigsum\tOR_list\tModel\n")
    for i in range(len(best)):
        f.write(str(i))
        f.write('\t')
        f.write(str(fitness[i][0]))
        f.write('\t')
        f.write(str(best[i].SOD))
        f.write('\t')
        f.write(str(best[i].igsum))
        f.write('\t')
        f.write(str(best[i].OR.tolist()))
        f.write('\t')
        f.write(str(best[i]))
        f.write('\n')

    f.close()
