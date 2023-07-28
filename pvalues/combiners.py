"""p-value combiners"""
import numpy as np
from scipy.stats import chi2

def fisher_pvalues_combiner(pvals):
    """Combine p-values using Fisher's method"""
    assert isinstance(pvals, np.ndarray), "Expected numpy array"
    n = len(pvals)
    score = -2 * np.sum(np.log(pvals))
    return chi2(2 * n).sf(score)

def min_pvalue_combiner(pvals):
    """Combine p-values using minimum, that is approximately Beta(1, n)"""
    assert isinstance(pvals, np.ndarray), "Expected numpy array"
    n = len(pvals)
    min_internal = min(pvals)
    return 1 - (1 - min_internal)**n
