"""Evaluate a graph given the model"""
# %% Imports
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import to_dense_adj
from gnn_data_generation.three_graph_families_dataset import ThreeGraphFamiliesDataset
from graph_vae import *
from pvalues.combiners import fisher_pvalues_combiner, min_pvalue_combiner

# %% Load model
model = torch.load(
    r"C:\Users\user\git\anomaly-detection\data\trained_on_gpu\230806c_vae_demo.pt",
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
    mega_sample = model.decoder(
        torch.stack([model.sampler(*encoded_ld) for _ in range(1000)]),
        torch.stack([model.sampler(*encoded_ud) for _ in range(1000)]))
    mega_sample = mega_sample.detach().mean(dim=[0]).sigmoid()

    # Probabilities of y_true
    y_true = to_dense_adj(datum.edge_index)[0]
    probs = y_true * mega_sample + (1 - mega_sample) * (1 - y_true)

    # Should be the same as
    bernoulli_dist = dist.Bernoulli(probs=mega_sample)
    probs_2 = bernoulli_dist.log_prob(y_true).exp()
    # assert equality
    assert torch.allclose(probs, probs_2)

    # AUC score
    auc_score = roc_auc_score(
        y_true.numpy().flatten(),
        mega_sample.numpy().flatten())
    print(f"AUC: {auc_score}")

    # I keep the probs only in the region that were one for either matrix
    locs = (mega_sample > 0.5).type(torch.int).logical_or(y_true)
    selected_probs = probs[locs]

    # Calculate p-values
    if len(selected_probs) > 0:
        fisher_score = fisher_pvalues_combiner(selected_probs.flatten().numpy())
        min_score = min_pvalue_combiner(selected_probs.flatten().numpy())
    else:
        fisher_score = 0
        min_score = 0

    if plots:
        print(f"Fisher p-value: {fisher_score}")
        print(f"Min p-value: {min_score}")

        n_true = y_true.sum().numpy().astype(int)
        n_pred = (mega_sample > 0.25).type(torch.int).sum().numpy().astype(int)
        n_common = y_true.logical_and((mega_sample > 0.25).type(torch.int)).sum().numpy().astype(int)
        print(f"Aimed {n_common}, missed {n_true - n_common}, misselected {n_pred - n_common}")

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(y_true)
        ax[0].set_title("True Adjacency")
        ax[1].imshow(mega_sample)
        ax[1].set_title("Model probability")
        ax[2].imshow((mega_sample > 0.25).type(torch.float))
        ax[2].set_title("Model probability thresholded")
        fig.tight_layout()
        fig.show()

    return fisher_score, min_score, auc_score

# %% Calculate for one example
selected_class = dataset[dataset.y == 0]
idx_choice = np.random.choice(len(selected_class))
test_datum = selected_class[idx_choice]
score_given_model(model, test_datum)

# %% Check for all
results = np.ndarray((0, 4))
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
plt.title("Fisher")
plt.show()

plt.hist(results[results[:, -1] == 0][:, 1], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.hist(results[results[:, -1] == 1][:, 1], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.hist(results[results[:, -1] == 2][:, 1], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.xlabel("p-value")
plt.legend(
    ("Data used to create the model",
     "Data from another process",
     "Yet another process"))
plt.title("Min p-value combiner")
plt.show()

plt.hist(results[results[:, -1] == 0][:, 2], density=True, alpha=0.3, range=[0.8, 1], bins=50)
plt.hist(results[results[:, -1] == 1][:, 2], density=True, alpha=0.3, range=[0.8, 1], bins=50)
plt.hist(results[results[:, -1] == 2][:, 2], density=True, alpha=0.3, range=[0.8, 1], bins=50)
plt.xlabel("AUC")
plt.legend(
    ("Data used to create the model",
     "Data from another process",
     "Yet another process"))
plt.title("AUC")

# %%
