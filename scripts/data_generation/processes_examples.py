"""Test data generation"""
# %% Imports
import time
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from IPython import display

def time_sequence_as_counting(_sequence):
    return np.cumsum(np.ones_like(_sequence))

def lin_law(x, a, b):
    return a + x * b

def power_law_fit(x, y):
    theta, _ = curve_fit(lin_law, np.log(x), np.log(y))
    return np.exp(theta[0]), theta[1]

def order_histogram(seq, *args, **kwargs):
    """Order histogram bins and fit quadratic law"""
    unique = np.unique(seq)
    counts = np.array([seq.count(i) for i in unique])
    order = np.argsort(counts)[::-1]
    dummy_x = np.arange(len(order))
    return plt.bar(dummy_x, counts[order], *args, **kwargs)

# %% Check Pitman-Yor data generation
from data_generation.data_generation import generate_pitman_yor
intensity = 7.0
discount = 0.25

t0 = time.time()
sequence = generate_pitman_yor(
    labels=50,
    intensity=intensity,
    discount=discount,
    length=1000,
    seed=0)
dt = time.time() - t0
print(f"Simulated in {dt}")

plt.hist(sequence, bins=50)
order_histogram(sequence, alpha=0.5)
plt.xticks(rotation = 75)
plt.title(f'Pitman-Yor samples distribution with intensity {intensity} discount {discount}')

# %% Anomalous PY
# Randomly generated anomalous base measure
from processes.dirichlet import sample_dirichlet_process_geometric_base
from processes.pitman_yor import generate_pitman_yor_with_anomaly
gen=np.random.default_rng(seed=0)

intensity = 7.0
discount = 0.25

masses = np.arange(50)
gen.shuffle(masses)
anomalous_base_measure = [
    masses,
    sample_dirichlet_process_geometric_base(intensity=1000, theta=0.05, n=50, gen=gen)]

t0 = time.time()
sequence = generate_pitman_yor_with_anomaly(
    labels=50,
    intensity=intensity,
    discount=discount,
    anomaly=np.array((*np.zeros(shape=500), *np.ones(shape=200), *np.zeros(shape=300))),
    anomaly_base_measure=anomalous_base_measure,
    length=1000,
    seed=0)
dt = time.time() - t0
print(f"Simulated in {dt}")

plt.hist(sequence)
plt.xticks(rotation = 75)
plt.title(f'Pitman-Yor samples distribution with intensity {intensity} discount {discount}')

# %% Check Poisson process data generation
from processes.poisson import poisson_process

rate = 1.
t0 = time.time()
time_sequence_0 = poisson_process(
    rate=rate,
    tmax=119,
    gen=0)
dt = time.time() - t0
print(f"Simulated in {dt}")

t0 = time.time()
time_sequence_1 = poisson_process(
    rate=0.4,
    tmax=119,
    gen=0)
dt = time.time() - t0
print(f"Simulated in {dt}")

try:
    time_sequence = poisson_process(
        rate=rate,
        length=100,
        tmax=118.1,
        gen=0)
except ValueError as e:
    print(e)

fig, ax = plt.subplots(1, 2)
ax[0].plot(time_sequence_0, time_sequence_as_counting(time_sequence_0))
ax[0].plot(time_sequence_1, time_sequence_as_counting(time_sequence_1))
ax[1].hist(np.diff(time_sequence_0));

# %% Test expected value of Poisson Point process
rate = 1.
max_time = 20
n_iter = 1000
final_ns = np.ndarray(shape=(0,))
for _ in range(n_iter):
    ts = poisson_process(
        rate=rate,
        tmax=max_time,
        gen=None)
    final_ns = np.append(final_ns, len(ts))

plt.hist(final_ns, alpha=0.5)
plt.vlines(x=rate * max_time, ymin=0, ymax=n_iter, colors=['red'])

# %% Test n independent Poisson Process
from processes.poisson import n_independent_poisson_processes

t0 = time.time()
time_sample, partition_sample = n_independent_poisson_processes(
    rates=[1., 0.4],
    tmax=119,
    gen=0
)
dt = time.time() - t0
print(f"Simulated in {dt}")

fig, ax = plt.subplots(1, 2)
ax[0].plot(
    time_sample[partition_sample == 0],
    time_sequence_as_counting(time_sample[partition_sample == 0]))
ax[0].plot(time_sample[partition_sample == 1],
         time_sequence_as_counting(time_sample[partition_sample == 1]))
ax[1].hist(np.diff(time_sample[partition_sample == 0]), alpha=0.5)
ax[1].hist(np.diff(time_sample[partition_sample == 1]), alpha=0.5);

# %% Test partitioned Poisson Process
from processes.poisson import partitioned_poisson_process

t0 = time.time()
time_sample, partition_sample = partitioned_poisson_process(
    rate=1.4,
    partition_probs=[2.5/3.5, 1/3.5],
    tmax=119,
    gen=0
)
dt = time.time() - t0
print(f"Simulated in {dt}")

fig, ax = plt.subplots(1, 2)
ax[0].plot(
    time_sample[partition_sample == 0],
    time_sequence_as_counting(time_sample[partition_sample == 0]))
ax[0].plot(time_sample[partition_sample == 1],
         time_sequence_as_counting(time_sample[partition_sample == 1]))
ax[1].hist(np.diff(time_sample[partition_sample == 0]), alpha=0.5)
ax[1].hist(np.diff(time_sample[partition_sample == 1]), alpha=0.5);

# %% Sinusoidal inhomogeneous Poisson Process
from processes.poisson import inhomogeneous_poisson_process_sinusoidal

rate0 = 1.
rate1 = 0.5
period = 4.
n = 15
# tmax = period * (1 + 2 * n) / 4
tmax = period * n / 2
n_iter = 1000
gen = None
time_sample_homo_end = np.ndarray(shape=(0,))
for _ in range(n_iter):
    simulate_homo = inhomogeneous_poisson_process_sinusoidal(
        rate0=rate0,
        rate1=0.,
        period=period,
        tmax=tmax,
        gen=gen
    )
    time_sample_homo_end = np.append(
        time_sample_homo_end,
        len(simulate_homo))
    
time_sample_end = np.ndarray(shape=(0,))
elapsed_time = np.ndarray(shape=(0,))
for _ in range(n_iter):
    t0 = time.time()
    simulate = inhomogeneous_poisson_process_sinusoidal(
        rate0=rate0,
        rate1=rate1,
        period=period,
        tmax=tmax,
        gen=gen
    )
    elapsed_time = np.append(elapsed_time, time.time() - t0)
    time_sample_end = np.append(
        time_sample_end,
        len(simulate))

fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(simulate_homo, time_sequence_as_counting(simulate_homo))
ax[0, 0].plot(simulate, time_sequence_as_counting(simulate))
ax[0, 0].legend(('Homogeneous', 'Inhomogeneous'))
ax[0, 0].set_xlabel('Time')
ax[0, 0].set_ylabel('Count')

ax[0, 1].plot(simulate_homo[1:], np.diff(simulate_homo))
ax[0, 1].plot(simulate[1:], np.diff(simulate))
ax[0, 1].set_xlabel('Time')
ax[0, 1].set_ylabel('Inter-arrival')

ax[1, 0].hist(time_sample_homo_end, alpha=0.5)
ax[1, 0].hist(time_sample_end, alpha=0.5)
ax[1, 0].legend(('Homogeneous', 'Inhomogeneous'))
ax[1, 0].vlines(x=rate0 * tmax + np.array([0, -rate1, rate1]),
                ymin=0,
                ymax=n_iter,
                linestyles=['solid', 'dashed', 'dashed'],
                colors=['red'])

ax[1, 1].hist(elapsed_time)
# %% Sample distribution from Dirichlet Process
from processes.dirichlet import sample_dirichlet_process_geometric_base
from scipy.stats import geom

NUM = 100
THETA = 0.05

dp_sample = sample_dirichlet_process_geometric_base(
    intensity=1000,
    theta=THETA,
    n=NUM,
    gen=np.random.default_rng(seed=0))

plt.plot(dp_sample, ls='None', marker='x')
plt.plot(geom(THETA).pmf(np.arange(1, NUM + 1)))
plt.xlabel('Support')
plt.ylabel('Mass')

# %% Simulate distance-dependent Chinese Restaurant Process
from processes.dirichlet import generate_exp_ddcrp
from processes.poisson import poisson_process

RATE = 1.
INTENSITY = 7.0
DECAY = 10.0
time_sequence = poisson_process(
    rate=RATE,
    tmax=100,
    gen=0)

t0 = time.time()
sequence = generate_exp_ddcrp(
    labels=50,
    intensity=INTENSITY,
    decay=DECAY,
    times=time_sequence,
    seed=0)
dt = time.time() - t0
print(f"Simulated in {dt}")

plt.hist(sequence, bins=50)
order_histogram(sequence, alpha=0.5)
plt.xticks(rotation = 75)
plt.title(f'DDCRP samples distribution with intensity {INTENSITY}')

# %%
