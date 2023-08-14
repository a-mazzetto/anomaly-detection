"""Evaluate a graph given the model"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from gnn_data_generation.three_graph_families_dataset import ThreeGraphFamiliesDataset
from gnn.vae import *

# %% Load model
model = torch.load(
    r"C:\Users\user\git\anomaly-detection\data\trained_on_gpu\230810_vae_demo_2.pt",
    map_location=torch.device('cpu'))

# %%
model.eval()

# %% Dataset
dataset = ThreeGraphFamiliesDataset(use_features=True)

# %% Calculate for one example
selected_class = dataset[dataset.y == 0]
idx_choice = np.random.choice(len(selected_class))
test_datum = selected_class[idx_choice]
vae_score_given_model(model, test_datum)

# %% Check for all
results = np.ndarray((0, 7))
MAX_ELEMENTS = 50
n = {0:0, 1:0, 2:0}
for idx, datum in enumerate(dataset):
    if n[int(datum.y)] < MAX_ELEMENTS:
        print(f"{idx}th element")
        results = np.vstack((results,
            np.array([*vae_score_given_model(model, datum, plots=False),
                    datum.y.item()]).reshape(1, -1))
        )
    n[int(datum.y)] += 1

plt.hist(results[results[:, -1] == 0][:, 0], density=True, alpha=0.3, range=[0.8, 1], bins=50)
plt.hist(results[results[:, -1] == 1][:, 0], density=True, alpha=0.3, range=[0.8, 1], bins=50)
plt.hist(results[results[:, -1] == 2][:, 0], density=True, alpha=0.3, range=[0.8, 1], bins=50)
plt.title("AUC")
plt.legend(
    ("Data used to create the model",
     "Data from another process",
     "Data from yet another process"))
plt.show()

plt.hist(results[results[:, -1] == 0][:, 1], density=True, alpha=0.3, color="blue")
plt.vlines(results[results[:, -1] == 0][:, 1].mean(), ymin=0, ymax=0.0012,
           linestyles="--", color="blue")
plt.hist(results[results[:, -1] == 1][:, 1], density=True, alpha=0.3, color="red")
plt.vlines(results[results[:, -1] == 1][:, 1].mean(), ymin=0, ymax=0.0012,
           linestyles="--", color="red")
plt.hist(results[results[:, -1] == 2][:, 1], density=True, alpha=0.3, color="green")
plt.vlines(results[results[:, -1] == 2][:, 1].mean(), ymin=0, ymax=0.0012,
           linestyles="--", color="green")
plt.title("log_prob")
plt.legend(
    ("Data used to create the model",
     "Data from another process",
     "Data from yet another process"))
plt.show()

# %%
