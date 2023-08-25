"""Test script to evaluate anomaly level in uthorization data"""
# %% Imports
import os
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from data_generation.constants import NNODES
from data_generation.dataset_operations import simple_aggregate_dataset_into_graphs
from data_generation.dataset_operations import torchdata_from_links_list
from gnn.vae import *
from gnn.train_loops import vae_training_loop, plot_vae_training_results
from gnn.early_stopping import EarlyStopping
from cuda.cuda_utils import select_device

# %% Data: load dataset as list of tuples and split into 15s graphs
DATASET = "./data/dataset_002/auth.txt"
dataset = []
with open(DATASET, "r", encoding="utf-8") as f:
    for line in f:
        time, source, destination, anomaly = line.strip().split("\t")
        dataset.append((float(time), source, destination, int(anomaly)))

# %% Divide into graphs with interval 15s
TIME_INTERVAL = 60
TRAINING_TIME = 24 * 3600
min_time = dataset[0][0]
max_time = dataset[-1][0]
# Number of graphs in the training period
n_training_graphs = int((TRAINING_TIME - min_time) // TIME_INTERVAL)
# Number of graphs which are anomalous
times = np.array([_d[0] for _d in dataset])
anomalies = np.array([_d[-1] for _d in dataset])
anomalous_times = times[np.where(anomalies)[0]]
anomalous_graphs = np.unique((anomalous_times - min_time) // TIME_INTERVAL).astype(int)
# Aggregate dataset: be aware that some graphs might be empty, n_graph is the index of non-empty graphs
aggregated_dataset, n_graph = simple_aggregate_dataset_into_graphs(
    dataset, TIME_INTERVAL)
# Create torch datasets
train_dataset = torchdata_from_links_list(
    [_agg for _agg, _n in zip(aggregated_dataset, n_graph) if _n <= n_training_graphs],
    max_nodes=NNODES)
deployment_dataset = torchdata_from_links_list(
    [_agg for _agg, _n in zip(aggregated_dataset, n_graph) if _n > n_training_graphs],
    max_nodes=NNODES)
deployment_dataset_anomaly = [_n in anomalous_graphs for _n in n_graph if _n > n_training_graphs],

# %% Model Training
train_idx, test_idx = torch.utils.data.random_split(
    train_dataset, (0.7, 0.3), torch.Generator().manual_seed(0))

model_train_dataset = [train_dataset[_idx] for _idx in train_idx.indices]
model_test_dataset = [train_dataset[_idx] for _idx in test_idx.indices]

train_loader = DataLoader(model_train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(model_test_dataset, batch_size=32)

device = select_device()

model = VAE(device=device, prior_modes=0, latent_dim=32, hidden_dim=64,
            n_gin_layers=3, graph_enc=False, pos_weight=None)
optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-4, weight_decay=0.5e-4)

history = vae_training_loop(model=model, optimizer=optimizer, num_epochs=2000,
    train_dl=train_loader, val_dl=test_loader, early_stopping=EarlyStopping(patience=100),
    best_model_path="./data/auth_static.pt", print_freq=10, device=device)

fig = plot_vae_training_results(history=history)
os.makedirs("./plots", exist_ok=True)
fig.savefig("./plots/auth_static.pdf")

# %%
model = torch.load("./data/auth_static/230810_auth_static.pt",
                   map_location=torch.device('cpu'))

model.eval()

# %%
scores = []
for datum in deployment_dataset:
    _, logp, pvalue, _, _, _, _ = vae_score_given_model(model, datum, plots=False)
    scores.append(logp)

anomalous_idxs = np.where(deployment_dataset_anomaly)
sorted_scores_args = np.argsort(scores)
# %%
