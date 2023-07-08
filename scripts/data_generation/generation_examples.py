"""Script containing all functions necessary for data generation"""
# %%
from copy import deepcopy
from typing import List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from processes.poisson import inhomogeneous_poisson_process_sinusoidal
from processes.pitman_yor import generate_pitman_yor

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
SEED = 0

# %% Generate times
times = inhomogeneous_poisson_process_sinusoidal(
    lambda_0, lambda_1, PERIOD, MAX_TIME, gen=SEED
)

# Look at results by hour
equidistant_times = np.arange(0, max(times), 1 * 60 * 60)
breakpoints = equidistant_times[:-1] + np.diff(equidistant_times)/2
histogram = np.histogram(times, bins=equidistant_times)
plt.plot(breakpoints, histogram[0], '-o')
plt.xlabel('Time')
plt.ylabel('Number of exvents per hour')
# %% Generate destinations

DESTINATION_DISCOUNT = 0.25
DESTINATION_INTENSITY = 7.0
destinations = generate_pitman_yor(
    discount=0.25,
    intensity=7.0,
    length=len(times),
    labels=computer_names,
    seed=SEED
)

plt.hist(destinations)
plt.xticks(rotation = 75)
plt.title(('Pitman-Yor samples distribution with intensity'
           f'{DESTINATION_INTENSITY} discount {DESTINATION_DISCOUNT}'))

# %% Generate sources conditional upon destination
link_dict = {}
for dest in np.unique(destinations):
    _n = destinations.count(dest)
    _sources = generate_pitman_yor(
        discount=0.25,
        intensity=7.0,
        length=_n,
        labels=computer_names,
        seed=SEED
    )
    link_dict[dest] = _sources

# %% Create links
link_dict_disposable = deepcopy(link_dict)
DISCRETE = True

dataset = []
for _time, _dest in zip(times, destinations):
    dataset.append((
        _time if not DISCRETE else np.floor(_time).astype(int),
        link_dict_disposable[_dest].pop(0),
        _dest))

# %% Write to file

with open('../../data/dataset.txt', 'w', encoding='utf-8') as file:
    for line in dataset:
        file.write(' '.join(str(s) for s in line) + '\n')

# %% Test equivalent function
from data_generation.data_generation import generate_dataset

NNODES  = 100
computer_names = [f'C{i}' for i in range(NNODES)]
# For example, simulate 12h
MAX_TIME = 12 * 60 * 60
# For example, 3h period
PERIOD = 3 * 60 * 60

def source_intensities(gen, size):
    return gen.choice(a=[2., 7., 12.], p=[0.2, 0.3, 0.5], size=size)

generate_dataset(
    max_time=MAX_TIME,
    period=PERIOD,
    n_nodes=NNODES,
    destination_intensity=7.0,
    destination_discount=0.25,
    source_intensities=source_intensities,
    source_discounts=0.5,
    node_names=computer_names,
    seed=0,
    dicretize_time=True,
    file_name="../../data/dataset_0.txt")

# %% Generate DDCRP
from data_generation.data_generation import generate_ddcrp_dataset

NNODES  = 100
computer_names = [f'C{i}' for i in range(NNODES)]
# For example, simulate 12h
MAX_TIME = 12 * 60 * 60
# For example, 3h period
PERIOD = 3 * 60 * 60
# 1h decay
DECAY = 1 * 60 * 60

def source_intensities(gen, size):
    return gen.choice(a=[2., 7., 12.], p=[0.2, 0.3, 0.5], size=size)

generate_ddcrp_dataset(
    max_time=MAX_TIME,
    period=PERIOD,
    n_nodes=NNODES,
    destination_intensity=7.0,
    destination_discount=0.25,
    source_intensities=source_intensities,
    source_decay=DECAY,
    node_names=computer_names,
    seed=0,
    discretize_time=True,
    file_name="../../data/dataset_0.txt")

# %%
