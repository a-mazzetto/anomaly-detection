"""Evaluate a graph given the model"""
# %% Imports
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import expit
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import to_dense_adj
from gnn_data_generation.three_graph_families_dataset import ThreeGraphFamiliesDataset
from graph_vae import *
from pvalues.combiners import fisher_pvalues_combiner, min_pvalue_combiner
from pvalues.auc_pvalue import auc_and_pvalue

# %% Load model
model = torch.load(
    r"C:\Users\user\git\anomaly-detection\data\trained_on_gpu\230808_vae_demo.pt",
    map_location=torch.device('cpu'))

# %%
model.eval()
# %% Dataset
dataset = ThreeGraphFamiliesDataset(use_features=True)

# %%
def score_given_model(model, datum, plots=True):
    """Assuming model of type Graph VAE"""
    # Bernoulli probabilities
    encoded_ld = model.encoder_ld(datum)
    encoded_ud = model.encoder_ud(datum)
    mega_logits = model.decoder(
        torch.stack([model.sampler(*encoded_ld) for _ in range(1000)]),
        torch.stack([model.sampler(*encoded_ud) for _ in range(1000)])).detach().numpy()
    mega_logits_mean = np.mean(mega_logits, axis=0)

    # Probabilities of y_true
    y_true = to_dense_adj(datum.edge_index)[0, ...].numpy()
    mega_logits_auc = roc_auc_score(y_true.flatten(), mega_logits_mean.flatten())
    auc_dist = np.ndarray(shape=(0,))
    for _ in range(1000):
        _index_sample = np.random.choice(1000, size=1000, replace=True)
        _mega_logits_mean = np.mean(mega_logits[_index_sample, ...], axis=0)
        auc_dist = np.append(
            auc_dist, roc_auc_score(y_true.flatten(), _mega_logits_mean.flatten()))
    pvalue = np.sum(auc_dist <= mega_logits_auc) / len(auc_dist)

    if plots:
        mega_mean = expit(mega_logits_mean)
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(y_true)
        ax[0].set_title("True Adjacency")
        ax[1].imshow(mega_mean)
        ax[1].set_title("Model probability")
        ax[2].imshow((mega_mean > 0.25).astype(float))
        ax[2].set_title("Model probability thresholded 0.25")
        fig.tight_layout()
        fig.show()

    return mega_logits_auc, pvalue

# %% Calculate for one example
selected_class = dataset[dataset.y == 0]
idx_choice = np.random.choice(len(selected_class))
test_datum = selected_class[idx_choice]
score_given_model(model, test_datum)

# %% Check for all
results = np.ndarray((0, 2))
MAX_ELEMENTS = 50
n = {0:0, 1:0, 2:0}
for idx, datum in enumerate(dataset):
    if n[int(datum.y)] < MAX_ELEMENTS:
        print(f"{idx}th element")
        results = np.vstack((results,
            np.array([*score_given_model(model, datum, plots=False),
                    datum.y.item()]).reshape(1, -1))
        )
    n[int(datum.y)] += 1

plt.hist(results[results[:, -1] == 0][:, 0], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.hist(results[results[:, -1] == 1][:, 0], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.hist(results[results[:, -1] == 2][:, 0], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.title("AUC")
plt.legend(
    ("Data used to create the model",
     "Data from another process",
     "Data from yet another process"))
plt.show()

plt.hist(results[results[:, -1] == 0][:, 1], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.hist(results[results[:, -1] == 1][:, 1], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.hist(results[results[:, -1] == 2][:, 1], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.title("p-value")
plt.legend(
    ("Data used to create the model",
     "Data from another process",
     "Data from yet another process"))
plt.show()
