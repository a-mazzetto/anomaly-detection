"""p-value calculation for poisson process"""
import numpy as np
from scipy.stats import poisson
from processes.pitman_yor_pvalue import PitmanYorPValue

class InhomogeneousPoissonPValue():
    """Calculate p-value using N(t_2) - N(t_1) distributed like
    Poisson(\int_{t_1}^{t_2} \lambda(s) ds)"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.thist = np.ndarray(shape=(0,))
        self.lamhist = np.ndarray(shape=(0,))

    def update_lambda(self, t, lam):
        """Update only the probability"""
        self.thist = np.append(self.thist, t)
        self.lamhist = np.append(self.lamhist, lam)

    def pvalue_and_update(self, t, lam):
        """Calculate p-value and update. The assumption is that only
        one event has been observed, hence N(t_2) - N(t_1) = 1"""
        if len(self.thist) == 0:
            self.update_lambda(t, lam)
            return 0.5
        else:
            dts = np.diff(np.append(self.thist, t))
            lams = np.append(self.lamhist, lam)
            mean_lams = (lams[:-1] + lams[1:]) / 2
            pois_lambda = np.dot(dts, mean_lams)
            # Reset sequence
            self.thist = np.array([t])
            self.lamhist = np.array([lam])
            # Return mid p-value
            cdf_right = poisson(pois_lambda).cdf(1)
            cdf_left = poisson(pois_lambda).cdf(0)
            return (cdf_left + cdf_right) / 2

class PitmanYorMarkedPPPValue():
    """p-value calculation for combined Poisson and PY processes for time
    and destination respectively"""
    def __init__(self, alpha, d, n_nodes):
        """Initialize processes"""
        self.time_processes = {}
        self.py_process = PitmanYorPValue(alpha, d, n_nodes)

    def reset(self):
        self.time_process = {}
        self.py_process.reset()

    def pvalue_and_update(self, t, lam, x):
        """Update probability for nodes different from x, initialize otherwise"""
        poisson_pvalue = None
        for xi, t_process in self.time_processes.items():
            p_xi = self.py_process.prob(xi)
            if xi != x:
                t_process.update_lambda(t, lam * p_xi)
            else:
                poisson_pvalue = t_process.pvalue_and_update(t, lam * p_xi)
        if poisson_pvalue is None:
            p_x = self.py_process.prob(x)
            self.time_processes[x] = InhomogeneousPoissonPValue()
            poisson_pvalue = self.time_processes[x].pvalue_and_update(t, lam * p_x)
        # Pitman-Yor p-value
        py_pvalue = self.py_process.pvalue_and_update(x)
        return py_pvalue, poisson_pvalue
