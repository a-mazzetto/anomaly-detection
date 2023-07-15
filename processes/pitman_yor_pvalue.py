"""Calculate p-value for Pitman-Yor process"""
from collections import Counter

class PitmanYorPValue():
    """Calculate Pitman-Yor p-value"""
    def __init__(self, alpha: float, d=float, n_nodes:int=0):
        "Initialize"
        self.alpha = alpha
        self.d = d
        self.n_nodes = n_nodes
        self.reset()

    def reset(self):
        "Reset"
        self.n = 0
        self.counter = Counter()
        self.kn = 0

    def update(self, x):
        """Add observation"""
        self.n += 1
        if x in self.counter:
            self.counter[x] += 1
        else:
            self.counter[x] = 1
        self.kn = len(self.counter)

    def prob(self, x):
        """Probability, useful for other applications"""
        prob_0 = (self.alpha + self.kn * self.d) / (self.alpha + self.n) / self.n_nodes if self.n_nodes > 0 else 0
        if x in self.counter:
            prob = prob_0 + (self.counter[x] - self.d) / (self.alpha + self.n)
        else:
            prob = prob_0
        return prob

    def _first_pvalue(self):
        """First p-value (n = 0)"""
        assert self.n == 0, "Enter this block only for the first p-value"
        right_pvalue = 1
        left_pvalue = 0
        return left_pvalue, right_pvalue

    def _unseen_pvalue(self, x):
        """p-value for unseen element"""
        assert x not in self.counter, "Enter only with unseen observations"
        assert self.n > 0, "Enter only with n > 0"
        term = (self.n_nodes - self.kn) / self.n_nodes if self.n_nodes > 0 else 1
        right_pvalue = (self.alpha + self.d * self.kn) / (self.alpha + self.n) * term
        left_pvalue = 0
        return left_pvalue, right_pvalue

    def _seen_pvalue(self, x):
        """p-value for alredy sen observation"""
        assert x in self.counter, "Observation expected to be seen already"
        nx = self.counter[x]
        # Right p-value first
        n_gt_nx = [n for n in self.counter.values() if n > nx]
        if self.n_nodes > 0:
            right_pvalue = 1 - (len(n_gt_nx) * (self.alpha + self.d * self.kn) / self.n_nodes + sum(n_gt_nx) - self.d * len(n_gt_nx)) / (self.alpha + self.n)
        else:
            right_pvalue = 1 - (sum(n_gt_nx) - self.d * len(n_gt_nx)) / (self.alpha + self.n)
        # Left p-value
        n_gteq_nx = n_gt_nx + [n for n in self.counter.values() if n == nx]
        if self.n_nodes > 0:
            left_pvalue = 1 - (len(n_gteq_nx) * (self.alpha + self.d * self.kn) / self.n_nodes + sum(n_gteq_nx) - self.d * len(n_gteq_nx)) / (self.alpha + self.n)
        else:
            left_pvalue = 1 - (sum(n_gteq_nx) - self.d * len(n_gteq_nx)) / (self.alpha + self.n)
        return left_pvalue, right_pvalue

    def pvalue(self, x):
        """Calculate p-value"""
        if self.n == 0:
            pvalues = self._first_pvalue()
        elif x not in self.counter:
            pvalues = self._unseen_pvalue(x)
        else:
            pvalues = self._seen_pvalue(x)
        return sum(pvalues) / 2  

    def pvalue_and_update(self, x):
        """Calcualte p-value, then update"""
        pvalue = self.pvalue(x)
        self.update(x)
        return pvalue

class StreamingPitmanYorPValue(PitmanYorPValue):
    """Extends PitmanYorPValue as a streaming version"""
    def __init__(self, twindow: float, alpha: float, d=float, n_nodes:int=0):
        super().__init__(alpha=alpha, d=d, n_nodes=n_nodes)
        self.twindow = twindow
        self.queue = []

    def reset(self):
        super().reset()
        self.queue = []

    def drop_outdated(self, t):
        """Removes old observations. Queue is (t, x)"""
        while len(self.queue) > 0 and self.queue[0][0] < (t - self.twindow):
            _, xold = self.queue.pop(0)
            # Update posterior accordingly
            self.n -= 1
            if self.counter[xold] == 1:
                _ = self.counter.pop(xold)
            else:
                self.counter[xold] -= 1
            self.kn = len(self.counter)

    def update(self, x, t):
        """Extends by dropping old observations first"""
        self.drop_outdated(t)
        self.queue.append((t, x))
        super().update(x=x)
    
    def pvalue(self, *_, **__):
        raise NotImplementedError(("p-value query without updating the posterior "
                                  "not implemented in the sequential case"))
    
    def _pvalue_internal(self, x):
        return super().pvalue(x)
    
    def pvalue_and_update(self, x, t):
        """Calcualte p-value, then update"""
        self.drop_outdated(t)
        pvalue = self._pvalue_internal(x)
        # Updating performs a second elimination of old samples, but will find none as the
        # postirior was already cleare of old observations
        self.update(x, t)
        return pvalue
