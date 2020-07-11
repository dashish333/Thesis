#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 02:23:48 2020

@author: ashishdwivedi
"""
import numpy as np
from l1solver import l1RegressionSolver
import copy
from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.utils.validation import check_random_state
import scipy.sparse as sp
from datetime import time
from sklearn.metrics.pairwise import euclidean_distances


# SET THE BELOW FIELDS ACCORDING THE YOUR EXPERIEMNT SETTING
sketch_size = []
l1normOpt = []
n = 0 # the number of rows in put feature matrix
row = n
iterations = 10
#####################################################

def initKPP(A,b):
    print("--initKPP--")
    l1norm = []  # this is for sketched
    ratio = []
    iter_data=[]
    time_per_k = []
    aug_A = np.c_[A,b]
    for k in sketch_size:
        r=0
        temp_data=[]
        temp_time = 0
        for i in range(iterations):
            X = np.c_[A,b]
            C=_k_init(X,n_clusters=k, x_squared_norms=row_norms(X,squared= True),random_state=None, n_local_trials=None)
            #print(C.shape)
            A_sketch = C[:,:-1] # placing the first d columns in A sketch
            b_sketch = C[:,-1][:,np.newaxis]  #placing the (d+1)th column in b
            #print(A_sketch.shape, b_sketch.shape)
            start = time.time()
            x_tilde = np.array(l1RegressionSolver(A_sketch,b_sketch))
            end = time.time()
            temp_time+=end-start
            regression_value=np.linalg.norm((A.dot(x_tilde)-b),ord=1)
            temp_data.append(regression_value)
            r+=regression_value
        time_per_k.append(temp_time/iterations)
        iter_data.append(temp_data)
        r/=iterations
        l1norm.append(r)
        # only for sketched matrix
    ratio = np.array(l1norm)/np.array(l1normOpt)
    print("")
    print("L1Norm = ",l1norm)
    print("Ratio = ",ratio)
    print("Time(secs) = ",time_per_k)
    print("---------------------\n")
    return ratio,l1norm,time_per_k

## -------------------------------
def _k_init(X, n_clusters, x_squared_norms, random_state=None, n_local_trials=None):
    """Init n_clusters seeds according to k-means++
    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).
    n_clusters : int
        The number of seeds to choose
    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape
    random_state = check_random_state(random_state)

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    assert x_squared_norms is not None, 'x_squared_norms None in _k_init'

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]

    return centers