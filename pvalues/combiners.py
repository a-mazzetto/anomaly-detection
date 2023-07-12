"""p-value combiners"""
import numpy as np
from scipy.stats import chi2

def fisher_pvalues_combiner(*args):
    """Combine p-values using Fisher's method"""
    n = len(args)
    score = -2 * np.sum(np.log(args))
    return chi2(2 * n).sf(score)

def min_pvalue_combiner(*args):
    """Combine p-values using minimum, that is approximately Beta(1, n)"""
    n = len(args)
    min_internal = min(args)
    return 1 - (1 - min_internal)**n
