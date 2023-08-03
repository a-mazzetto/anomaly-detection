"""Parameter estimation with DDCRP"""
# %%
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from processes.dirichlet import generate_exp_ddcrp
from processes.poisson import poisson_process
from parameter_estimation.parameter_estimation import \
    dirichlet_true_kn, dirichlet_est_pars

# %% Initial parameters

N_NODES = 100
N_ITERS = 1000
N = 10000
TRUE_ALPHA = 5.0

true_kn = dirichlet_true_kn(TRUE_ALPHA, N, N_NODES)

# %% Simulate processes and estimate Kn and H1n
meas_kn_list = []
meas_alpha_list = []
for _ in range(N_ITERS):
    times = poisson_process(
        rate=10.0,
        length=N
    )
    sequence = generate_exp_ddcrp(
        intensity=TRUE_ALPHA,
        decay=1.0,
        times=times,
        labels=N_NODES
    )
    counter = Counter(sequence)
    meas_kn = len(counter)
    meas_alpha = dirichlet_est_pars(meas_kn, N, N_NODES)
    meas_kn_list.append(meas_kn)
    meas_alpha_list.append(meas_alpha)

# %%
fig, ax = plt.subplots(1, 2)
fig.tight_layout()
ax[0].hist(meas_alpha_list, alpha=0.5)
ax[0].vlines(x=TRUE_ALPHA, ymin=0, ymax=N_ITERS/4, linestyle="--", color="red")
ax[0].set_xlabel(r"$\alpha$")
ax[0].set_ylabel("Density")

ax[1].hist(meas_kn_list, alpha=0.5)
ax[1].vlines(x=true_kn, ymin=0, ymax=N_ITERS/4, linestyle="--", color="red")
ax[1].set_xlabel(r"$K_n$")
ax[1].set_ylabel("Density")

# %%

fig.savefig("./plots/ddcrp_param_est.pdf")
