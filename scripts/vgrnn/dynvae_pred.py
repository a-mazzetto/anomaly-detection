"""Evaluate a graph given the model"""
# %% Imports
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import unbatch, unbatch_edge_index
from gnn_data_generation.vgrnn_test_dataset import VGRNNTestDataset
from gnn.dynvae import *
from pvalues.combiners import fisher_pvalues_combiner, min_pvalue_combiner

# %% Load model
model = torch.load(
    r"C:\Users\user\git\anomaly-detection\data\trained_on_gpu\dynvae_10_epocs.pt",
    map_location=torch.device('cpu'))

# %%
model.eval()

def evaluate_model(model, datum):
    y_true = to_dense_adj(datum.edge_index, batch=datum.batch)
    res = model(
        torch.stack(unbatch(datum.x, batch=datum.batch)),
        unbatch_edge_index(datum.edge_index, batch=datum.batch),
        y_true
    )
    return y_true, [torch.sigmoid(i) for i in res[-1]]

# %% Dataset
dataset = VGRNNTestDataset(use_features=True)

# %%
def score_given_model(model, datum, plots=True):
    """Assuming model of type Graph VAE"""
    # Bernoulli probabilities
    y_pred = None
    for _ in range(100):
        y_true, yi_pred = evaluate_model(model, datum)
        if y_pred is None:
            y_pred = [_t.detach().expand(1, -1, -1) for _t in yi_pred]
        else:
            for idx in range(len(y_pred)):
                y_pred[idx] = torch.cat(
                    (y_pred[idx], yi_pred[idx].detach().expand(1, -1, -1)))
    mega_sample = [_t.mean(dim=[0]) for _t in y_pred]

    # Probabilities of y_true
    probs = [_y_true * _mega_sample + (1 - _mega_sample) * (1 - _y_true) for
             _y_true, _mega_sample in zip(y_true, mega_sample)]

    # AUC score
    auc_score = [roc_auc_score(_y_true.numpy().flatten(),
                               _mega_sample.numpy().flatten()) for
                               _y_true, _mega_sample in zip(y_true, mega_sample)]
    print(f"AUC: {auc_score}")

    # I keep the probs only in the region that were one for either matrix
    locs = [(_mega_sample > 0.5).type(torch.int).logical_or(_y_true) for
            _y_true, _mega_sample in zip(y_true, mega_sample)]
    selected_probs = [_probs[_locs] for _probs, _locs in zip(probs, locs)]

    # Calculate p-values
    if len(selected_probs) > 0:
        fisher_score = fisher_pvalues_combiner(torch.cat(selected_probs).flatten().numpy())
        min_score = min_pvalue_combiner(torch.cat(selected_probs).flatten().numpy())
    else:
        fisher_score = 0
        min_score = 0

    if plots:
        print(f"Fisher p-value: {fisher_score}")
        print(f"Min p-value: {min_score}")

        n_true = [_y_true.sum().numpy().astype(int) for _y_true in y_true]
        n_pred = [(_mega_sample > 0.25).type(torch.int).sum().numpy().astype(int) for
                  _mega_sample in mega_sample]
        n_common = [_y_true.logical_and((_mega_sample > 0.25).type(torch.int)).sum().numpy().astype(int) for
                    _y_true, _mega_sample in zip(y_true, mega_sample)]
        print((f"Aimed {[i.item() for i in n_common]}, missed {[i - j for i, j in zip(n_true, n_common)]}, "
              f"misselected {[i - j for i, j in zip(n_pred, n_common)]}"))

        fig, ax = plt.subplots(len(y_pred), 3)
        for i in range(len(y_pred)):
            ax[i, 0].imshow(y_true[i])
            ax[i, 1].imshow(mega_sample[i])
            ax[i, 2].imshow((mega_sample[i] > 0.25).type(torch.float))
            if i == 0:
                ax[i, 0].set_title("True Adjacency")
                ax[i, 1].set_title("Model probability")
                ax[i, 2].set_title("Model probability thresholded")
        fig.set_figheight(10)
        fig.tight_layout()
        fig.show()

    return fisher_score, min_score

# %% Calculate for one example
idx_choice = np.random.choice(len(dataset))
y_true, y_pred = evaluate_model(model, dataset[idx_choice])
score_given_model(model, dataset[idx_choice])

# %% Check for all
results = np.ndarray((0, 3))
for idx, datum in enumerate(dataset):
    print(f"{idx}th element")
    results = np.vstack((results,
        np.array([*score_given_model(model, datum, plots=False),
                  datum.y.item()]).reshape(1, -1))
    )

plt.hist(results[results[:, -1] == 0][:, 0], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.hist(results[results[:, -1] == 1][:, 0], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.title("Fisher")
plt.show()

plt.hist(results[results[:, -1] == 0][:, 1], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.hist(results[results[:, -1] == 1][:, 1], density=True, alpha=0.3, range=[0, 1], bins=50)
plt.xlabel("p-value")
plt.legend(
    ("Data used to create the model",
     "Data from another process"))
plt.title("Min p-value combiner")

# %%
