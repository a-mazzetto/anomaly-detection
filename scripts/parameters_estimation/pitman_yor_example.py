"""Parameter estimation for Pitman-Yor process"""
# %%
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from processes.pitman_yor import generate_pitman_yor
from parameter_estimation.parameter_estimation import \
    pitman_yor_true_kn_h1n, pitman_yor_est_pars

# %% Initial parameters

N_NODES = 100
N_ITERS = 10000
N = 1000
TRUE_ALPHA = 5.0
TRUE_D = 0.25

true_kn, true_h1n = pitman_yor_true_kn_h1n(TRUE_ALPHA, TRUE_D, N, N_NODES)

# %% Simulate processes and estimate Kn and H1n
meas_kn_list = []
meas_h1n_list = []
meas_alpha_list = []
meas_d_list = []
for _ in range(N_ITERS):
    sequence = generate_pitman_yor(
        discount=TRUE_D,
        intensity=TRUE_ALPHA,
        length=N,
        labels=N_NODES
    )
    counter = Counter(sequence)
    meas_kn = len(counter)
    meas_h1n = sum(np.equal(list(counter.values()), 1))
    meas_alpha, meas_d = pitman_yor_est_pars(meas_kn, meas_h1n, N, N_NODES)
    meas_kn_list.append(meas_kn)
    meas_h1n_list.append(meas_h1n)
    meas_alpha_list.append(meas_alpha)
    meas_d_list.append(meas_d)

# %%
fig, ax = plt.subplots(2, 2)
ax[0, 0].hist(meas_alpha_list, alpha=0.5)
ax[0, 0].vlines(x=TRUE_ALPHA, ymin=0, ymax=N_ITERS/4, linestyle="--", color="red")
ax[0, 0].set_xlabel(r"$\alpha$")
ax[0, 0].set_ylabel("Density")

ax[0, 1].hist(meas_d_list, alpha=0.5)
ax[0, 1].vlines(x=TRUE_D, ymin=0, ymax=N_ITERS/4, linestyle="--", color="red")
ax[0, 1].set_xlabel(r"$d$")

ax[1, 0].hist(meas_kn_list, alpha=0.5)
ax[1, 0].vlines(x=true_kn, ymin=0, ymax=N_ITERS/4, linestyle="--", color="red")
ax[1, 0].set_xlabel(r"$K_n$")
ax[1, 0].set_ylabel("Density")

ax[1, 1].hist(meas_h1n_list, alpha=0.5)
ax[1, 1].vlines(x=true_h1n, ymin=0, ymax=N_ITERS/4, linestyle="--", color="red")
ax[1, 1].set_xlabel(r"$H_{1n}$")
fig.tight_layout()

# %%

fig.savefig("./plots/py_par_est.pdf")
