"""Evaluate rank the dataset given the model using Schulze method"""
# %% Imports
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
import schulze
from gnn.dynvae import DynVAE, dynvae_score_given_model
from data_generation.kpath_dyngraphs import kpath_dyngraphs

# %% User Input
MODEL = "./data/trained_on_gpu/dynvae_part2_model_modes.pt"
DATASET = "./data/dataset_003/auth_ddcrp.txt"
SCORES = "./data/dataset_003/DP/phase_4_source_score.txt"
INTERVAL = 15 * 60
N_INTERVALS = 4
K_PATH_LEN = 3

OUTPUT_FILE = "./data/dataset_003/final_ranking.txt"

# %% Load model
model = torch.load(
    MODEL,
    map_location=torch.device('cpu'))
model.eval()

# %% Evaluate on target nodes
nodes = []
part1_scores = []
part2_scores = []
with open(SCORES, "r", encoding="utf-8") as file:
    for line in file:
        source, score, _ = line.strip().split("\t")
        node_dynamic_graph = kpath_dyngraphs(
            dataset_file=DATASET,
            scores_file=SCORES,
            interval=INTERVAL,
            n_intervals=N_INTERVALS,
            k_path_len=K_PATH_LEN,
            target_node=source)

        _, norm_logps, _ = dynvae_score_given_model(
            model,
            node_dynamic_graph,
            plots=False,
            norm_log_prob=True)
        norm_logp = np.mean(norm_logps)

        print(f"Node {source}, score {norm_logp}")
        nodes.append(source)
        part1_scores.append(score)
        part2_scores.append(norm_logp)
nodes = np.array(nodes)
part1_scores = np.array(part1_scores)
part2_scores = np.array(part2_scores)
# Part 1: the smallest the score, the highest the anomaly level
# Part 2: the highest the normalised p-value, the worse
part1_rank = nodes[np.argsort(part1_scores)]
part2_rank = nodes[np.argsort(part2_scores)[::-1]]

# %% Final rank with Schulze methods's

d = defaultdict(int)
schulze._add_ranks_to_d(d=d, ranks=[[i] for i in part1_rank], weight=1)
schulze._add_ranks_to_d(d=d, ranks=[[i] for i in part2_rank], weight=1)
final_rank = schulze._rank_p(nodes, schulze._compute_p(d, nodes))
final_rank_expanded = [j for i in final_rank for j in i]

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    file.writelines([i + "\n" for i in final_rank_expanded])
