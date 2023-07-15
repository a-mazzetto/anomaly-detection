"""Example of Poisson p-value calculation"""

# %% Imports
import numpy as np
from scipy.signal import square
import matplotlib.pyplot as plt
from processes.poisson import inhomogeneous_poisson_process_sinusoidal
from processes.poisson_pvalue import InhomogeneousPoissonPValue

# %% Generate process
rate0 = 15.
rate1 = 1.
period = 1.
n = 10
tmax = period * n / 2

simulate = inhomogeneous_poisson_process_sinusoidal(
    rate0=rate0,
    rate1=rate1,
    period=period,
    tmax=tmax,
    gen=0
)

real_lambda = rate0 + rate1 * np.sin(2 * np.pi / period * simulate)
t_scan = np.arange(0, max(simulate), 0.01)
real_lambda_scan = rate0 + rate1 * np.sin(2 * np.pi / period * t_scan)

plt.plot(t_scan, real_lambda_scan)
plt.plot(simulate, real_lambda, "o")

# %%
# Note that we should have all 0.5, but the fact that we are integrating
# a square function makes it not perfect
pvalue = InhomogeneousPoissonPValue()
pvalues = np.ndarray(shape=(0,))

for t, lam in zip(simulate, real_lambda):
    pvalues = np.append(pvalues, pvalue.pvalue_and_update(t, lam))

pvalue_skipping = InhomogeneousPoissonPValue()
pvalues_skipping = np.ndarray(shape=(0,))

for t, lam in zip(simulate, real_lambda):
    if np.random.uniform() > 0.5:
        pvalues_skipping = np.append(pvalues_skipping,
            pvalue_skipping.pvalue_and_update(t, lam))
    else:
        pvalues_skipping = np.append(pvalues_skipping, np.nan)
        pvalue_skipping.update_lambda(t, lam)

plt.hist(pvalues, color="blue", alpha=0.3, density=True)
plt.hist(pvalues_skipping, color="red", alpha=0.3, density=True)
plt.xlabel("Mid p-value")
plt.ylabel("Density")

# %%
