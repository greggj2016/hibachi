import numpy as np
import pandas as pd
from copy import deepcopy as COPY
from itertools import combinations as combos
from itertools import product as prod
from time import time
import pdb

def get_joint_probs(X, y, N, P, subset_lists):
    data = np.concatenate((X, y), axis = 1)
    unique_Xval_lists = np.array([np.unique(X[:, i]).tolist() for i in range(P)], dtype = object)
    unique_val_lists = np.array([np.unique(data[:, i]).tolist() for i in range(P + 1)], dtype = object)
    p_y = np.sum(y == np.unique(y), axis = 0)/len(y)
    joint_prob_sets = []
    joint_pseudoprob_sets = []
    for i in range(len(subset_lists)):
        y_index = len(unique_val_lists) - 1
        y_indices = y_index*np.ones((len(subset_lists[i]), 1))
        subset_list = subset_lists[i]
        full_subset_list = np.concatenate((y_indices, subset_lists[i]), axis = 1).astype(int)    
        X_subsets = X[:, subset_lists[i]]
        unique_val_sets = unique_val_lists[full_subset_list]
        unique_val_sets = [list(prod(*pair)) for pair in unique_val_sets]
        unique_Xval_sets = unique_val_lists[subset_list]
        unique_Xval_sets = [list(prod(*pair)) for pair in unique_Xval_sets]
        k_way_joint_probs = []
        k_way_joint_pseudoprobs = []
        for k in range(len(unique_val_sets)):
            uv_set = np.array(unique_val_sets[k])
            uXv_set = np.array(unique_Xval_sets[k])
            X_subset = X_subsets[:, k, :]
            data_subset = np.concatenate((y, X_subset), axis = 1)
            instances = (data_subset == uv_set.reshape(len(uv_set), 1, len(uv_set[0])))
            probs = np.sum(np.all(instances, axis = 2), axis = 1)/N
            k_way_joint_probs.append(probs)
            Xinstances = (X_subset == uXv_set.reshape(len(uXv_set), 1, len(uXv_set[0])))
            Xprobs = np.sum(np.all(Xinstances, axis = 2), axis = 1)/N
            pseudoprobs = np.outer(p_y, Xprobs).reshape(-1) 
            k_way_joint_pseudoprobs.append(pseudoprobs)
        joint_prob_sets.append(k_way_joint_probs)
        joint_pseudoprob_sets.append(k_way_joint_pseudoprobs)
    return(joint_prob_sets, joint_pseudoprob_sets)

def compute_MI(X, y):
    N, P = len(X), len(X[0]) 
    subset_lists = [np.array(list(combos(range(P), i + 1))) for i in range(P)]
    joint_prob_sets, joint_pseudoprob_sets = get_joint_probs(X, y, N, P, subset_lists)
    MI_sets = []
    for i in range(len(joint_prob_sets)):
        joint_probs = joint_prob_sets[i]
        joint_pseudoprobs = joint_pseudoprob_sets[i]
        MI_vals = []
        for k in range(len(joint_probs)):
            p = joint_probs[k].astype(float)
            p_pseudo = joint_pseudoprobs[k].astype(float)
            MI_log_part = np.log2((p + 1E-20)/(p_pseudo + 1E-20))
            MI_vals.append(np.sum(p*MI_log_part))
        MI_sets.append(MI_vals)
    MI_sets_old = COPY(MI_sets)
    for i in range(len(MI_sets)):
        for k in range(i):
            old_MI_dim = len(subset_lists[k][0])
            this_subset = np.array( old_MI_dim*[subset_lists[i]]).transpose((1, 0, 2))
            broadcasting_dims = (len(subset_lists[k]), 1,  old_MI_dim, 1)
            old_subset = subset_lists[k].reshape(broadcasting_dims)
            related_subsets = np.all(np.any(old_subset == this_subset, axis = 3), axis = 2)
            redundant_info = np.sum(MI_sets[k]*related_subsets.T, axis = 1)
            MI_sets[i] -= redundant_info
    return(MI_sets)
