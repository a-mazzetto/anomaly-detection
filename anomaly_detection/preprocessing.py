"""Script to estimate parameters from dataset"""
# %%
import os
from collections import Counter
import numpy as np
from scipy.stats import beta
import matplotlib.pylab as plt
from input_parameters import *
from parameter_estimation.parameter_estimation import pitman_yor_est_pars, dirichlet_est_pars

TSTART = PREPROCESSING_TSTART
TEND = PREPROCESSING_TEND
THERSHOLD = PREPROCESSING_THERSHOLD

destination_counter = Counter()
source_counters = {}
with open(FILE_PATH, "r", encoding="utf-8") as file:
    for line in file:
        time, source, dest, _ = line.strip().split("\t")
        time = float(time)
        if time >= TSTART and time <= TEND:
            if dest not in destination_counter:
                destination_counter[dest] = 1
                source_counters[dest] = Counter()
                source_counters[dest][source] = 1
            else:
                destination_counter[dest] += 1
                if source not in source_counters[dest]:
                    source_counters[dest][source] = 1
                else:
                    source_counters[dest][source] += 1
        if time > TEND:
            break

# Estimate destination process parameters
destination_process_params = pitman_yor_est_pars(
    meas_kn=len(destination_counter),
    meas_h1n=sum(np.isclose(list(destination_counter.values()), 1)),
    n=sum(destination_counter.values()),
    n_nodes=N_NODES)

# Estimate source processes parameters
dest_list = list(source_counters.keys())
dest_alpha = np.nan * np.ones(len(dest_list))
dest_d = np.nan * np.ones_like(dest_alpha)
for i, dest in enumerate(dest_list):
    counter = source_counters[dest]
    if sum(counter.values()) > THERSHOLD:
        if DDCRP:
            dest_alpha[i] = dirichlet_est_pars(
                meas_kn=len(counter),
                n=sum(counter.values()),
                n_nodes=N_NODES)
        else:
            dest_alpha[i], dest_d[i] = pitman_yor_est_pars(
                meas_kn=len(counter),
                meas_h1n=sum(np.isclose(list(counter.values()), 1)),
                n=sum(counter.values()),
                n_nodes=N_NODES)

fig, ax = plt.subplots(1, 2)
ax[0].hist(dest_alpha, density=True)
ax[0].set_xlabel(r"$\alpha$")
ax[0].vlines(x=2.0, ymin=0, ymax=0.6,
             linestyle="--", color="red")
ax[0].vlines(x=7.0, ymin=0, ymax=0.3,
             linestyle="--", color="red")
ax[0].vlines(x=12.0, ymin=0, ymax=0.1,
             linestyle="--", color="red")
if not np.all(np.isnan(dest_d)):
    ax[1].hist(dest_d, density=True)
    x_plot = np.linspace(0, 1, 100)
    ax[1].plot(x_plot, beta(a=2.0, b=5.0).pdf(x_plot), linestyle="--", color="red")
    ax[1].set_xlabel("$d$")
fig.savefig(os.path.join(RESULTS_FOLDER, "preprocessing.pdf"))

# Fill NAN
dest_alpha[np.isnan(dest_alpha)] = np.nanmedian(dest_alpha)
if not np.all(np.isnan(dest_d)):
    dest_d[np.isnan(dest_d)] = np.nanmedian(dest_d)

# Save data to file
y_parameters_file = os.path.join(RESULTS_FOLDER, "y_params.txt")
x_given_y_parameters_file = os.path.join(RESULTS_FOLDER, "x_given_y_params.txt")

with open(y_parameters_file, "w", encoding="utf-8") as file:
    file.write("\t".join([str(i) for i in destination_process_params]) + "\n")

with open(x_given_y_parameters_file, "w", encoding="utf-8") as file:
    if not np.all(np.isnan(dest_d)):
        for line in zip(dest_list, dest_alpha.astype(str), dest_d.astype(str)):
            file.write("\t".join(line) + "\n")
    else:
        for line in zip(dest_list, dest_alpha.astype(str)):
            file.write("\t".join(line) + "\n")

# %%
