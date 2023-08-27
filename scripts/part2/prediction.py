"""Evaluate a graph given the model"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from gnn.dynvae import DynVAE, dynvae_score_given_model
from data_generation.kpath_dyngraphs import kpath_dyngraphs

# %% Load model
model = torch.load(
    "./data/trained_on_gpu/dynvae_part2_model_modes.pt",
    map_location=torch.device('cpu'))

# %%
model.eval()

# %% Dataset
dataset = kpath_dyngraphs(
    dataset_file="./data/dataset_003/auth_ddcrp.txt",
    scores_file="./data/dataset_003/DP/phase_4_source_score.txt",
    max_t=24 * 60 * 60,
    interval=15 * 60,
    n_intervals=4,
    k_path_len=3)

# %% Calculate for one example
idx_choice = np.random.choice(len(dataset))
dynvae_score_given_model(model, dataset[idx_choice])

# %% Check for all
results = np.ndarray((0, 6))
MAX_ELEMENTS = 50
idx = 0
while idx < min(MAX_ELEMENTS, len(dataset)):
    print(f"{idx}th element")
    auc, logp, pvalue, conf = dynvae_score_given_model(
        model, dataset[idx], plots=False)
    results = np.vstack((results,
        np.array([
            np.mean(auc),
            np.mean(logp),
            *np.mean(conf, axis=0)]).reshape(1, -1))
    )
    idx += 1

# %% Evaluate on target nodes

nodes = []
scores = []
with open("./data/dataset_003/DP/phase_4_source_score.txt", "r", encoding="utf-8") as file:
    for line in file:
        source, _, _ = line.strip().split("\t")
        node_dynamic_graph = kpath_dyngraphs(
            dataset_file="./data/dataset_003/auth_ddcrp.txt",
            scores_file="./data/dataset_003/DP/phase_4_source_score.txt",
            interval=15 * 60,
            n_intervals=4,
            k_path_len=3,
            target_node=source)

        _, norm_logp, pvalue, _ = dynvae_score_given_model(
            model,
            node_dynamic_graph,
            plots=False,
            norm_log_prob=True)

        print(f"Node {source}, score {norm_logp}, pvalue {pvalue}")
        nodes.append(source)
        scores.append(norm_logp)

# %%
