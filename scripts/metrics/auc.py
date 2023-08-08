"""Understanding of relationship between AUC and Mann-Whitney U-test"""
# %% Imports
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from pvalues.auc_pvalue import auc_and_pvalue

# %% Data
true = np.hstack((np.ones(20), np.zeros(80)))
pred = np.random.uniform(size=100)
# rank_pred = pred[np.argsort(pred)[::-1]]
# %% Metrics
roc_auc_score(true, pred)

n1 = np.sum(true)
n0 = len(true) - n1

order = np.argsort(pred)
rank = np.argsort(order)
rank += 1
U1 = np.sum(rank[true == 1]) - n1 * (n1 + 1)/2
# U0 = np.sum(rank[true == 0]) - n0 * (n0 + 1)/2
AUC1 = U1/ (n1 * n0)
# AUC0 = U0/ (n1*n0)

# Recall that np.sort() is in descending order: for the best ranking we expect small ranks for `true == 0`
# and high ranks for `true == 1`. If x is the `true == 1` and y is the `true == 0` sample and F(u) and G(u)
# are the corresponding distributions, than we have that the best case is with F(u) > G(u)
# Null hypothesis: no difference in the central tendency of the two rankings
# Alternative hypothesis:
#   "two-sided": there is difference in the central tendency of the two rankings
#   "less": F(u) > G(u) 
#   "greater": F(u) < G(u)
ustat, pval = mannwhitneyu(
    rank[true == 1], rank[true == 0],
    alternative="greater", use_continuity=False)
print(ustat / (n1 * n0))
print(pval)

# %% Implementation
auc_and_pvalue(true, pred)

# %%
