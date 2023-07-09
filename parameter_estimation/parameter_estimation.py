"""Functions to estimate parameters"""
import numpy as np
from scipy.special import gamma
from scipy.optimize import fsolve

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

def pitman_yor_est_pars(meas_kn, meas_h1n, n, n_nodes):
    """Parameter estimation of Pitman-Yor parameters following the dissertation presented in
    ``Modelling Dynamic Network Evolution as Pitman-Yor Process" by Passino and Heard. In
    particular, we use here only the best estimation technique.
    """
    # Adjustment under the Birthday Problem
    kn_hat = np.log(1 - meas_kn / n_nodes) / np.log((n_nodes - 1) / n_nodes)
    h1n_hat = meas_h1n * (n_nodes / (n_nodes - 1))**(kn_hat - 1)
    # Estimate parameters under approximation `Method 2` proposed in the paper
    d_hat = h1n_hat / kn_hat
    def _equation(x):
        return d_hat * gamma(d_hat + x) * kn_hat - gamma(1 + x) * n**d_hat
    alpha_hat = fsolve(_equation, kn_hat / np.log(n))[0]
    return alpha_hat, d_hat
