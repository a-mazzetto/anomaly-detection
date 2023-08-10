"""Test script to evaluate anomaly level in uthorization data"""
# %% Imports
import numpy as np
from data_generation.constants import NNODES
from data_generation.dataset_operations import simple_aggregate_dataset_into_graphs
from data_generation.dataset_operations import torchdata_from_links_list

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

# %%
