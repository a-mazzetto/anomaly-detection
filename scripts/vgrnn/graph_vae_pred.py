"""Evaluate a graph given the model"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from gnn_data_generation.three_graph_families_dataset import ThreeGraphFamiliesDataset
from graph_vae import *

# %% Load model
model = torch.load(
    r"C:\Users\user\git\anomaly-detection\data\trained_on_gpu\vae_demo_simple_2.pt",
    map_location=torch.device('cpu'))

# %%
model.eval()
# %% Dataset
dataset = ThreeGraphFamiliesDataset(use_features=True)

# %% Predict
selected_class = dataset[dataset.y == 0]
idx_choice = np.random.choice(len(selected_class))
test_datum = selected_class[idx_choice]
_, _, y_pred, y_true = model(test_datum)
# %%
# Use Poisson Binomial Distribution
import torch.distributions as dist
mega_sample = model.decoder(
    dist.Normal(*model.encoder_ld(test_datum)).sample((1000,)),
    dist.Normal(*model.encoder_ud(test_datum)).sample((1000,)))
mega_sample = mega_sample.mean(dim=[0])
y_pred_bin = (y_pred > 0.5).type(torch.float)
probs = y_pred_bin * mega_sample + (1 - mega_sample) * (1 - y_pred_bin)

# Should be the same as
bernoulli_dist = dist.Bernoulli(probs=mega_sample)
probs_2 = bernoulli_dist.log_prob((y_pred > 0.5).type(torch.float)).exp()
# assert equality
torch.allclose(probs, probs_2)

# Calculate united p-value
from pvalues.combiners import fisher_pvalues_combiner, min_pvalue_combiner

print(f"Fisher p-value: {fisher_pvalues_combiner(probs.flatten().numpy())}")
print(f"Min p-value: {min_pvalue_combiner(probs.flatten().numpy())}")
# %% Plot adjacency matrices
fig, ax = plt.subplots(1, 2)
ax[0].imshow(y_true[0])
ax[0].set_title("True Adjacency")
ax[1].imshow(y_pred_bin[0])
ax[1].set_title("Model Adjacency")

# %%
n_true = y_true[0].sum().numpy().astype(int)
n_pred = y_pred_bin[0].sum().numpy().astype(int)
n_common = y_true[0].logical_and(y_pred_bin[0]).sum().numpy().astype(int)
print(f"Aimed {n_common}, missed {n_true - n_common}, misselected {n_pred - n_common}")

# %%
