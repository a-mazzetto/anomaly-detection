"""Functions to estimate parameters"""
import numpy as np
from scipy.special import gamma, gammaln
from scipy.optimize import fsolve, root

def pitman_yor_true_kn_h1n(true_alpha, true_d, n, n_nodes):
    """Parameter estimation of Pitman-Yor parameters following the dissertation presented in
    ``Modelling Dynamic Network Evolution as Pitman-Yor Process" by Passino and Heard
    """
    true_h1n = (gamma(1 + true_alpha)  * n**true_d) / gamma(true_d + true_alpha)
    true_kn = true_h1n / true_d
    # Correct for discrete base measures
    true_kn_corr = n_nodes * (1 - ((n_nodes - 1) / n_nodes)**true_kn)
    true_h1n_corr = true_h1n * ((n_nodes - 1) / n_nodes)**(true_kn - 1)
    return true_kn_corr, true_h1n_corr

def kn_h1n_hat(meas_kn, meas_h1n, n_nodes):
    """Adjustment of kn and h1n under the Birthday Problem
    """
    # Adjustment under the Birthday Problem
    kn_hat = np.log(1 - meas_kn / n_nodes) / np.log((n_nodes - 1) / n_nodes)
    h1n_hat = meas_h1n * (n_nodes / (n_nodes - 1))**(kn_hat - 1)
    return kn_hat, h1n_hat

def kn_hat(meas_kn, n_nodes):
    """Adjustment of kn and h1n under the Birthday Problem
    """
    # Adjustment under the Birthday Problem
    kn_hat = np.log(1 - meas_kn / n_nodes) / np.log((n_nodes - 1) / n_nodes)
    return kn_hat

def pitman_yor_est_pars(meas_kn, meas_h1n, n, n_nodes, logaritmic=True):
    """Parameter estimation of Pitman-Yor parameters following the dissertation presented in
    ``Modelling Dynamic Network Evolution as Pitman-Yor Process" by Passino and Heard. In
    particular, we use here only the best estimation technique.
    """
    # Adjustment under the Birthday Problem
    kn_hat = np.log(1 - meas_kn / n_nodes) / np.log((n_nodes - 1) / n_nodes)
    h1n_hat = meas_h1n * (n_nodes / (n_nodes - 1))**(kn_hat - 1)
    # Estimate parameters under approximation `Method 2` proposed in the paper
    initial_guess = [kn_hat / np.log(n), h1n_hat / kn_hat]
    def _equation(x):
        _a, _d = x[0], x[1]
        if _a > 0 and _d > 0 and _d < 1:
            if logaritmic:
                return [gammaln(1 + _a) - gammaln(_d + _a) + _d * np.log(n) - np.log(_d) - np.log(kn_hat + _a / _d),
                        (h1n_hat - _a) / kn_hat - _d]
            else:
                return [h1n_hat * gamma(_d + _a) - gamma(1 + _a) * n**_d,
                        h1n_hat - _a - kn_hat * _d]
        else:
            return [-100, -100]
    root_res = root(_equation, initial_guess)
    if root_res.success:
        alpha_hat, d_hat = root_res.x
    else:
        alpha_hat, d_hat = initial_guess
    return alpha_hat, d_hat

def dirichlet_true_kn(true_alpha, n, n_nodes):
    """Parameter estimation of Pitman-Yor parameters following the dissertation presented in
    ``Modelling Dynamic Network Evolution as Pitman-Yor Process" by Passino and Heard
    """
    true_kn = true_alpha * np.log(n)
    # Correct for discrete base measures
    true_kn_corr = n_nodes * (1 - ((n_nodes - 1) / n_nodes)**true_kn)
    return true_kn_corr

def dirichlet_est_pars(meas_kn, n, n_nodes):
    """Parameter estimation of Pitman-Yor parameters following the dissertation presented in
    ``Modelling Dynamic Network Evolution as Pitman-Yor Process" by Passino and Heard. In
    particular, we use here only the best estimation technique.
    """
    # Adjustment under the Birthday Problem
    kn_hat = np.log(1 - meas_kn / n_nodes) / np.log((n_nodes - 1) / n_nodes)
    # Estimate parameters under approximation `Method 2` proposed in the paper
    initial_guess = kn_hat / np.log(n)
    def _equation(_a):
        if _a > 0:
            return _a * np.log(1 + n / _a) - kn_hat
        else:
            return -100
    root_res = root(_equation, initial_guess)
    if root_res.success:
        alpha_hat = root_res.x
    else:
        alpha_hat = initial_guess
    return alpha_hat

class ForgettingFactorsMean():
    """Forgetting Factors estimate of Mean"""
    def __init__(self, lam, keep_hist=False, num_0=0, den_0=0):
        """Forgetting Factor Initialization"""
        assert lam >=0 and lam <= 1, "Lambda must be in [0, 1]"
        self.lam = lam
        self.keep_hist = keep_hist
        self.reset(num_0, den_0)

    def reset(self, num_0=0, den_0=0):
        """Reset"""
        self.num = num_0
        self.den = den_0
        self.n = 0
        if self.keep_hist:
            self.hist = np.ndarray(shape=(0,))
        else:
            self.hist = None
            

    def update(self, num, den):
        """Update"""
        if den <= 0:
            den += 1e-8
        assert den > 0, "Denominator must be positive"
        self.num = self.lam * self.num + num
        if self.den > 0:
            self.den = self.lam * self.den + den
        else:
            self.den = den
        self.n += 1
        if self.keep_hist:
            self.hist = np.append(self.hist, self.mean)
    
    @property
    def mean(self):
        return self.num / self.den

def poisson_process_lam_est(
        sequence,
        interval=None,
        forg_factor=1,
        num_0=0,
        den_0=0):
    """Lambda parameter estimation for Poisson Process.
    
    :param sequence: time sequence
    :param interval: aggregation interval
    :parameter forg_factor: forgetting factors weight (set to 1 for
                            normal mean)"""
    if interval is not None:
        counts, bins = np.histogram(
            sequence,
            bins = np.arange(min(sequence), max(sequence), 1))
    else:
        if np.any(np.diff(sequence) == 0):
            raise ValueError("Coincident times. Consider specifying interval")
        counts = np.ones_like(sequence[1:])
        bins = sequence
    lam = ForgettingFactorsMean(lam=forg_factor, keep_hist=True)
    lam.reset(num_0=num_0, den_0=den_0)
    for _idx, _count in enumerate(counts):
        lam.update(_count, bins[_idx + 1] - bins[_idx])
    return bins[1:], lam.hist

class PoissonProcessRateEstimation():
    """Forgetting factors estimate of lambda for PP"""
    def __init__(self, forgetting_factor, num_0=0, den_0=0, t_start=0):
        """Initialize forgetting factors mean estimate, and time zero"""
        self.mean = ForgettingFactorsMean(lam=forgetting_factor, num_0=num_0, den_0=den_0)
        self.num_0 = num_0
        self.den_0 = den_0
        self.t_start = t_start
        self.reset()

    def reset(self):
        """Reset"""
        self.mean.reset(self.num_0, self.den_0)
        self.t_old = self.t_start

    def update(self, t):
        """Update numerator and denominator using $\lambda(t) = \frac{1}{n \delta}
        \sum_{i=1}^n N_i(l(t), l(t) + \delta)$ in its streaming version"""
        self.mean.update(1, t - self.t_old)
        self.t_old = t

    @property
    def rate_est(self):
        return self.mean.mean
