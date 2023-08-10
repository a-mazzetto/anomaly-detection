"""Operations on datasets"""
from typing import List
import numpy as np
import torch
from torch_geometric.data import Data

def join_datasets_and_sort(dataset_0, dataset_1):
    """Assuming the first column is time"""
    dataset = dataset_0 + dataset_1
    times = np.array([i[0] for i in dataset]).astype(float)
    reordering = np.argsort(times)
    return [dataset[i] for i in reordering]

def simple_aggregate_dataset_into_graphs(dataset:List[tuple], time_interval:float,
                                         rename:bool=True) -> List[np.ndarray]:
    """Aggregate by time interval. Using only (time, source, destination) data when the
    incoming dataset is expected to be of the form (time, source, destination, anomaly).

    Parameters:
    :param List[tuple] dataset: dataset in format (time, source, destination, anomaly)
    :param float time_interval: aggregation interval
    :param bool rename: if `True`, each graph's nodes are renamed
    :return List[np.ndarray] graphs_by_links: list of graph links
    """
    times, sources, destinations, _ = list(zip(*dataset))
    links = np.hstack((np.array(sources).reshape(-1, 1), np.array(destinations).reshape(-1, 1)))
    aggregated_times = np.array(times) // time_interval
    aggregated_times = aggregated_times.astype(int)
    n_intervals = int(times[-1] // time_interval)
    graphs_by_links = []
    n_graph = []
    for i in range(n_intervals):
        if np.any(aggregated_times == i):
            idxstart, idxend = np.where(aggregated_times == i)[0][[0, -1]]
            if idxend > idxstart:
                unique_links = np.unique(links[idxstart:idxend], axis=0)
                unique_nodes = np.unique(unique_links)
                if rename:
                    mapping = {index: i for i, index in enumerate(unique_nodes)}
                    mapping_vec = np.vectorize(lambda x: mapping[x])
                    renamed_unique_links = mapping_vec(unique_links)
                    graphs_by_links.append(renamed_unique_links)
                else:
                    graphs_by_links.append(unique_links)
            n_graph.append(i)
    return graphs_by_links, n_graph

def torchdata_from_links_list(links_list, features_list=None, max_nodes=None):
    if features_list is None and max_nodes is None:
            raise ValueError("Either `features_list` or `max_nodes` must be provided!")
    n_features = max_nodes if features_list is None else features_list[0].shape[-1]
    data_list = []
    for n_graph, links in enumerate(links_list):
        n_nodes = links.max() + 1
        if features_list is not None:
            _feat = features_list[n_graph]
            assert _feat.shape == (n_nodes, n_features), "Unexpected features matrix dimension"
            features = torch.tensor(_feat, dtype=torch.float).reshape(n_nodes, -1)
        else:
            features = torch.nn.ConstantPad1d((0, max_nodes - n_nodes), 0.)(torch.eye(n_nodes))
        links_matrix = torch.tensor(links, dtype=torch.int64).t().contiguous()
        geom_data = Data(x=features, edge_index=links_matrix)
        assert geom_data.validate(), "Issues with Data creation"
        data_list.append(geom_data)
    return data_list
