"""Calculate p-value for Pitman-Yor process"""
from collections import Counter
import numpy as np

class DDCRPPValue():
    """Calculate distance-dependent CRP p-value"""
    def __init__(self, alpha: float, beta: float, n_nodes:int=0):
        """Initialize DDCRP with exponential decay function with constant Beta"""
        self.alpha = alpha
        self.beta = beta
        self.beta = beta
        self.n_nodes = n_nodes
        self.reset()

    def reset(self):
        "Reset"
        self.n = 0
        self.t0 = 0
        self.counter = Counter()
        self.kn = 0

    def update(self, x, t):
        """Add observation"""
        self.t0 = t if self.n == 0 else self.t0
        self.n += 1
        if x in self.counter:
            self.counter[x] += np.exp((t - self.t0) / self.beta)
        else:
            self.counter[x] = np.exp((t - self.t0) / self.beta)
        self.kn = len(self.counter)

    def _first_pvalue(self):
        """First p-value (n = 0)"""
        assert self.n == 0, "Enter this block only for the first p-value"
        right_pvalue = 1
        left_pvalue = 0
        return (left_pvalue + right_pvalue) / 2

    def _unseen_pvalue(self, x):
        """p-value for unseen element"""
        assert x not in self.counter, "Enter only with unseen observations"
        assert self.n > 0, "Enter only with n > 0"
        term = (self.n_nodes - self.kn) / self.n_nodes if self.n_nodes > 0 else 1
        right_pvalue = self.alpha / (self.alpha + self.n) * term
        left_pvalue = 0
        return (left_pvalue + right_pvalue) / 2

    def _seen_pvalue(self, x):
        """p-value for alredy sen observation"""
        assert x in self.counter, "Observation expected to be seen already"
        etot = sum(self.counter.values())
        ex = self.counter[x]
        # Right p-value first
        e_gt_ex = [e for e in self.counter.values() if e > ex]
        if self.n_nodes > 0:
            pvalue = 1 - (len(e_gt_ex) * self.alpha / self.n_nodes + self.n * sum(e_gt_ex) / etot) / (self.alpha + self.n)
        else:
            pvalue = 1 - (self.n * sum(e_gt_ex) / etot) / (self.alpha + self.n)
        return pvalue

    def pvalue(self, x):
        """Calculate p-value"""
        if self.n == 0:
            pvalue = self._first_pvalue()
        elif x not in self.counter:
            pvalue = self._unseen_pvalue(x)
        else:
            pvalue = self._seen_pvalue(x)
        return pvalue 

    def pvalue_and_update(self, x, t):
        """Calcualte p-value, then update"""
        pvalue = self.pvalue(x)
        self.update(x, t)
        return pvalue
