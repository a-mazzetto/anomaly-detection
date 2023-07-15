"""p-value calculation for poisson process"""
import numpy as np
from scipy.stats import poisson

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
            return np.nan
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
