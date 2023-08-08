"""p-value calculation using: S. J. MASON and N. E. GRAHAM `Areas beneath the relative operating characteristics
(ROC) and relative operating levels (ROL) curves: Statistical signi cance and interpretation`"""

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

def auc_and_pvalue(y_true, y_pred):
    """p-value calculation for AUC.
    Recall that np.sort() is in descending order: for the best ranking we expect small ranks for `true == 0`
    and high ranks for `true == 1`. If x is the `true == 1` and y is the `true == 0` sample and F(u) and G(u)
    are the corresponding distributions, than we have that the best case is with F(u) > G(u)
    Null hypothesis: no difference in the central tendency of the two rankings
    Alternative hypothesis:
    "two-sided": there is difference in the central tendency of the two rankings
    "less": F(u) > G(u) 
    "greater": F(u) < G(u)"""
    auc_check = roc_auc_score(y_true, y_pred)
    # Number of 1 and 0
    n1 = np.sum(y_true)
    n0 = len(y_true) - n1
    # Create ranking (descending)
    order = np.argsort(y_pred)
    rank = np.argsort(order)
    rank += 1
    # Calculate Mann-Whitney Test. Less alternative, as we want to have: worse AUC, smaller p-value
    ustat, pval = mannwhitneyu(
        rank[y_true == 1], rank[y_true == 0],
        alternative="less", use_continuity=False)
    assert np.isclose(ustat / (n1 * n0), auc_check), "These two should be the same!"
    return auc_check, pval
