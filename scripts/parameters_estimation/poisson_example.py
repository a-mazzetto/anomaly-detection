"""Parameter estimation for inhomogeneous Poisson Point Process
The increment N(t + h) - N(t) has Poisson distribution with mean
\int_t^{t + h} \lambda(s) ds"""
# %% Imports
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from processes.poisson import inhomogeneous_poisson_process_sinusoidal
from parameter_estimation.parameter_estimation import poisson_process_lam_est

# %%
rate0 = 10.
rate1 = 2.
period = 200.
tmax = 600
n_iter = 1000
gen = None

process = inhomogeneous_poisson_process_sinusoidal(
    rate0=rate0,
    rate1=rate1,
    period=period,
    tmax=tmax,
    gen=gen
)
# %%
lam_bins, lam_hist = poisson_process_lam_est(process, None, 0.97)

plt.plot(lam_bins, rate0 + rate1 * np.sin(2 * np.pi / period * lam_bins))
plt.plot(lam_bins, lam_hist)
# %%
