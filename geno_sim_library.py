import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import multivariate_normal as mnorm
from itertools import combinations as combos
from itertools import product
from scipy.stats import pearsonr
import pdb


# applying section 3, equation 7, from the following document:
# http://science.oregonstate.edu/~madsenl/files/MadsenBirkes2013.pdf
class F:
    
    def __init__(self, maf, A = np.array([0,1,2])):
        self.A = A
        self.maf = maf
        self.p = np.array([(1 - maf)**2, 2*maf*(1 - maf), maf**2])
        self.P = np.round(np.cumsum(self.p), 6)
        self.mu = np.sum(A*self.p)
        self.sig = np.sum(((A - self.mu)**2)*self.p)**(1/2)
        
    def cdf(self, a):
        ind = np.searchsorted(self.A, a, side = "right") - 1
        return(self.P[ind])

    def inv_cdf(self, u):
        ind = np.searchsorted(self.P, u)
        return(self.A[ind])

def compute_R(F1, F2, delta):
    geno_pairs = np.array(list(product([0,1,2], [0,1,2]))).T
    F1_vals = F1.cdf(geno_pairs[0])
    F2_vals = F2.cdf(geno_pairs[1])
    mu = np.array([0,0])
    inv_norm_vals = np.array([norm.ppf(F1_vals), norm.ppf(F2_vals)]).T
    ones = np.ones(len(inv_norm_vals))
    cov_max = np.array([[1, delta],[delta, 1]])
    copula_vals_max = mnorm(mu, cov_max).cdf(inv_norm_vals)
    summation_max = np.sum(ones - F1_vals - F2_vals + copula_vals_max)
    R_pred = (summation_max - F1.mu*F2.mu)/(F1.sig*F2.sig)
    return(R_pred)

def get_delta_loss(delta, args):
    F1, F2, R, TS = args
    R_pred = compute_R(F1, F2, delta[0])
    return((R - R_pred)**2)

def get_delta(F1, F2, R):
    R_max = compute_R(F1, F2, 1 - 1E-6)
    guess = R
    if R > R_max:
        message = "exiting: the specified correlation of " + str(R)
        message += " is too high for SNPs with minor allele "
        message += "frequencies " + str(F1.maf) + " and " + str(F2.maf)
        message += ". The maximum possible R is " + str(R_max) + "."
        print(message)
        exit()
    results = minimize(get_delta_loss, R, [F1, F2, R, False],
                       bounds = [(1E-6, 1 - 1E-6)])
    if results.success:
        return(results.x, R_max)
    else:
        message = "exiting: optimization failed to converge. "
        message = "Try rerunning the code. Please otify the Github "
        message = "author if this happens repeatedly or consistently. "
        print(message)
        exit()

def simulate_correlated_SNPs(mafs, R_vec, N):
    F_set = [F(maf) for maf in mafs]
    F_pairs = list(combos(F_set, 2))
    maf_pairs = list(combos(mafs, 2))
    cov_z_info = np.zeros(np.sum(np.arange(len(mafs))))
    R_max = np.zeros(np.sum(np.arange(len(mafs))))
    cov_z = np.zeros((len(mafs), len(mafs)))
    for i in range(len(F_pairs)):
        F1, F2 = F_pairs[i]
        R = R_vec[i]
        cov_z_info[i], R_max[i] = get_delta(F1, F2, R)

    covariance_info = pd.DataFrame(maf_pairs)
    covariance_info.columns = ["maf1", "maf2"]
    covariance_info["R_input"] = R_vec
    covariance_info["R_max"] = R_max
    covariance_info.to_csv("covariance_info.txt", sep = "\t", header = True, index = False)

    cov_z[np.triu_indices(len(mafs), 1)] = cov_z_info
    cov_z += 0.5*np.eye(len(mafs))
    cov_z += cov_z.T
    if np.any(np.linalg.eigvals(cov_z) < 0):
        message = "\n\nExiting: seed covariance matrix is not positive definite. "
        message += "One or more SNP correlations are too close to their "
        message += "maximum possible correlations. Either reduce input R "
        message += "values of decrease maf differences between SNP pairs.\n\n"

        message += "Given your current input mafs, a list of your input R values "
        message += "next to their corresponding maximum values was created. " 
        message += "Reduce the input R values that are closest to their maxima. "
        message += "Alternatively, make a SNP pair's mafs closer in order to "
        message += "increase the maximum possible R value between the SNPs."
        print(message)
        exit()

    Z = np.random.multivariate_normal(np.zeros(len(mafs)), cov_z, N)
    U = norm.cdf(Z)
    X = np.array([F_set[i].inv_cdf(U[:, i]) for i in range(len(F_set))])
    print(np.corrcoef(X))
    return(X.T)

'''
mafs = [0.5]*10
R_vec = [0.5]*np.sum(np.arange(10)) 
simulate_correlated_SNPs(mafs, R_vec, 10000000)
'''