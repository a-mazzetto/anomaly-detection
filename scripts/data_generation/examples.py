"""Test data generation"""
# %% Imports
import time
import matplotlib.pyplot as plt

from data_generation.data_generation import generate_pitman_yor

# %% Check Pitman-Yor data generation
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

# %%
