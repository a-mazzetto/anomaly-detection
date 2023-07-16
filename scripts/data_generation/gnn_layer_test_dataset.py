"""Generate dataset for test of GNN convolution layers"""
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from data_generation import constants
from data_generation.data_generation import generate_dataset
from data_generation.dataset_operations import simple_aggregate_dataset_into_graphs

TIME_INTERVAL = 15 * 60

# %% First group of data
def random_source_intensities(gen, size):
    """Most computers (60%) see routinary activity"""
    return gen.choice(a=[2., 7., 12.], p=[0.6, 0.3, 0.1], size=size)

first_group_data = []

for _ in range(5):
    dataset, _ = generate_dataset(
        max_time=2 * constants.MAX_TIME,
        period=2 * constants.PERIOD,
        n_nodes=constants.NNODES,
        destination_intensity=7.0,
        destination_discount=0.25,
        source_intensities=random_source_intensities,
        source_discounts=0.25,
        node_names=constants.NNODES,
        seed=None,
        discretize_time=False,
        file_name=None)

    first_group_data.extend(simple_aggregate_dataset_into_graphs(dataset, TIME_INTERVAL))

# %% Second group data

second_group_data = []

for _ in range(5):
    dataset, _ = generate_dataset(
        max_time=2 * constants.MAX_TIME,
        period=2 * constants.PERIOD,
        n_nodes=constants.NNODES,
        destination_intensity=7.0,
        destination_discount=0.25,
        source_intensities=7.0,
        source_discounts=0.25,
        node_names=constants.NNODES,
        seed=None,
        discretize_time=False,
        file_name=None)

    second_group_data.extend(simple_aggregate_dataset_into_graphs(dataset, TIME_INTERVAL))

# %% Third group data

third_group_data = []

for _ in range(5):
    dataset, _ = generate_dataset(
        max_time=2 * constants.MAX_TIME,
        period=2 * constants.PERIOD,
        n_nodes=constants.NNODES,
        destination_intensity=7.0,
        destination_discount=0.75,
        source_intensities=7.0,
        source_discounts=0.75,
        node_names=constants.NNODES,
        seed=None,
        discretize_time=False,
        file_name=None)

    third_group_data.extend(simple_aggregate_dataset_into_graphs(dataset, TIME_INTERVAL))

# %% Visualize some graphs
from igraph import Graph, plot as igplot

fig, ax = plt.subplots(2, 2)

for i in range(4):
    graph = Graph.TupleList(
        first_group_data[i],
        directed=True,
        weights=True)
    igplot(graph, target=ax[i // 2, i % 2], layout="auto")
# %%
