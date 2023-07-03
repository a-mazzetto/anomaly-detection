"""Test data generation"""
# %% Imports
import time
import numpy as np
import matplotlib.pyplot as plt

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

plt.hist(sequence)
plt.xticks(rotation = 75)
plt.title(f'Pitman-Yor samples distribution with intensity {intensity} discount {discount}')

# %% Check Poisson process data generation
from data_generation.data_generation import generate_poisson_process

rate = 1.
t0 = time.time()
time_sequence_0 = generate_poisson_process(
    rate=rate,
    length=100,
    seed=0)
dt = time.time() - t0
print(f"Simulated in {dt}")

t0 = time.time()
time_sequence_1 = generate_poisson_process(
    rate=rate,
    max=118.1,
    seed=0)
dt = time.time() - t0
print(f"Simulated in {dt}")

try:
    time_sequence = generate_poisson_process(
        rate=rate,
        length=100,
        max=118.1,
        seed=0)
except ValueError as e:
    print(e)

fig, ax = plt.subplots(1, 2)
ax[0].plot(time_sequence_0)
ax[0].plot(time_sequence_1)
ax[1].hist(np.diff(time_sequence_0))

# %%
