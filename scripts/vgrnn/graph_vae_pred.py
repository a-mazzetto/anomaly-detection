"""Evaluate a graph given the model"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from gnn_data_generation.three_graph_families_dataset import ThreeGraphFamiliesDataset
from graph_vae import *

# %% Load model
model = torch.load(
    r"C:\Users\user\git\anomaly-detection\data\trained_on_cpu\vae_demo_2_modes.pt",
    map_location=torch.device('cpu'))

# %%
model.eval()
# %% Dataset
dataset = ThreeGraphFamiliesDataset(use_features=True)

# %% Predict
selected_class = dataset[dataset.y == 0]
idx_choice = np.random.choice(len(selected_class))
test_datum = selected_class[idx_choice]
_, _, _, y_true = model(test_datum)

# %%
# Use Poisson Binomial Distribution
import torch.distributions as dist
mega_sample = model.decoder(
    dist.Normal(*model.encoder_ld(test_datum)).sample((1000,)),
    dist.Normal(*model.encoder_ud(test_datum)).sample((1000,)))
mega_sample = mega_sample.mean(dim=[0])
# Probability of y_true given the model
probs = y_true * mega_sample + (1 - mega_sample) * (1 - y_true)

# Should be the same as
bernoulli_dist = dist.Bernoulli(probs=mega_sample)
probs_2 = bernoulli_dist.log_prob(y_true).exp()
# assert equality
print(torch.allclose(probs, probs_2))

# I keep the probs only in the regione that were one for either matrix
locs = (mega_sample > 0.5).type(torch.int).logical_or(y_true)
selected_probs = probs[0][locs[0]]

# Calculate united p-value
from pvalues.combiners import fisher_pvalues_combiner, min_pvalue_combiner

print(f"Fisher p-value: {fisher_pvalues_combiner(selected_probs.flatten().numpy())}")
print(f"Min p-value: {min_pvalue_combiner(selected_probs.flatten().numpy())}")
# %% Plot adjacency matrices
fig, ax = plt.subplots(1, 3)
ax[0].imshow(y_true[0])
ax[0].set_title("True Adjacency")
ax[1].imshow(mega_sample)
ax[1].set_title("Model probability")
ax[2].imshow((mega_sample > 0.25).type(torch.float))
ax[2].set_title("Model probability thresholded")

# %%
n_true = y_true[0].sum().numpy().astype(int)
n_pred = (mega_sample > 0.25).type(torch.int).sum().numpy().astype(int)
n_common = y_true[0].logical_and((mega_sample > 0.25).type(torch.int)).sum().numpy().astype(int)
print(f"Aimed {n_common}, missed {n_true - n_common}, misselected {n_pred - n_common}")

# %%
