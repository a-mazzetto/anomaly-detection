"""Example of p-value calculation"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from processes.poisson import inhomogeneous_poisson_process_sinusoidal
from processes.pitman_yor import generate_pitman_yor
from processes.pitman_yor_pvalue import PitmanYorPValue, StreamingPitmanYorPValue

# %% Generate dataset
NNODES  = 100
computer_names = [f'C{i}' for i in range(NNODES)]
# For example, simulate 12h
MAX_TIME = 12 * 60 * 60
# For example, 3h period
PERIOD = 3 * 60 * 60
# From LANL dataset: 0.8-1.4M/4h connections with 12000 nodes
lambda_min_max = np.array((0.8, 1.4)) * 1e6 / 12000 / (4 * 60 * 60) * NNODES
lambda_0 = np.mean(lambda_min_max)
lambda_1 = np.diff(lambda_min_max)[-1] / 2

times = inhomogeneous_poisson_process_sinusoidal(
    lambda_0, lambda_1, PERIOD, MAX_TIME, gen=0
)
destinations = generate_pitman_yor(
    discount=0.25,
    intensity=7.0,
    length=len(times),
    labels=computer_names,
    seed=0
)

# %% Sequential p-value calculation
pvalue = PitmanYorPValue(alpha=7.0, d=0.25, n_nodes=NNODES)
pvalues = np.ndarray(shape=(0,))
for _x in destinations:
    pvalues = np.append(pvalues, pvalue.pvalue_and_update(_x))

# %% Streaming p-value calculation
stream_pvalue = StreamingPitmanYorPValue(twindow=3 * 60 * 60, alpha=7.0, d=0.25, n_nodes=NNODES)
stream_pvalues = np.ndarray(shape=(0,))
for _t, _x in zip(times, destinations):
    stream_pvalues = np.append(stream_pvalues, stream_pvalue.pvalue_and_update(_x, _t))

plt.hist(pvalues, alpha=0.2, color="blue")
plt.hist(stream_pvalues, alpha=0.2, color="red")

# %%
