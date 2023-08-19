"""Dynamic graph generation"""
# %% Imports
import pandas as pd
import igraph as ig
import torch
from torch_geometric.data import Data, Batch
from anomaly_detection.utils import switched_open

# %% Main Function
def kpath_dyngraphs(
        dataset_file,
        scores_file,
        max_t = 1e8,
        interval = 60,
        n_intervals = 4,
        k_path_len = 3,
        min_nodes = 10,
        target_node = None
):
    """Script to generate dynamic graphs from a dataset of type (time, source, destimation, anomaly) for training
    of a dynamic graph variational autoencoder
    :param dataset_file: dataset of type (time, source, destimation, anomaly)
    :param scores_file: source nodes with scores from first part of anomaly detection
    :param max_t: max training time (disregard dataset after)
    :param interval: aggregation time interval
    :param n_intervals: number of intervals in the dynamic graphs
    :param k_path_len: length of out-path links used to construct subgraphs
    :param min_nodes: minimum number of nodes to consider the graph
    :param target_node: switches to extraction of dynamic graph for specific node"""
    if target_node is not None and max_t != 1e8:
        print("When target node is specified, max time input is disregarded")
    source_nodes = pd.read_table(scores_file, names=("score", "timewhen"), index_col=0,
                                 dtype={"score": float, "timewhen": float})
    if target_node is not None:
        target_time = source_nodes.loc[target_node].timewhen
        min_t = target_time + (0.5 - n_intervals) * interval
        max_t = target_time + 0.5 * interval
    else:
        min_t = 0
    # Create graphs
    macro_graphs = []
    current_links_list = []
    subgraph_list = []
    current_t0 = None
    current_group = None
    with switched_open(dataset_file, "r") as file:
        for line in file:
            time, source, destination, _ = line.strip().split("\t")
            time = float(time)
            if time > max_t:
                break
            if time > min_t:
                macro_group = (time - min_t) // (interval * n_intervals)
                if current_group is None:
                    current_group = macro_group
                if current_t0 is None:
                    current_t0 = time - min_t
                if macro_group != current_group:
                    graph = ig.Graph.TupleList(current_links_list, directed=True, weights=False)
                    graph.vs["score"] = source_nodes.loc[graph.vs["name"]].score.tolist()
                    graph.es["subgraph"] = subgraph_list
                    macro_graphs.append(graph)
                    current_group = macro_group
                    current_links_list = []
                    subgraph_list = []
                    current_t0 = time - min_t
                if source in source_nodes.index:
                    subgraph_list.append(int((time - min_t - current_t0) // interval))
                    current_links_list.append((source, destination))
    # Discharge last graph
    if len(current_links_list) > 0:
        graph = ig.Graph.TupleList(current_links_list, directed=True, weights=False)
        graph.vs["score"] = source_nodes.loc[graph.vs["name"]].score.tolist()
        graph.es["subgraph"] = subgraph_list
        macro_graphs.append(graph)
    # Extract graphs
    subgraphs = []
    for _node in source_nodes.index:
        if target_node is not None and _node != target_node:
            continue
        for _graph in macro_graphs:
            if _graph.vs.select(name=_node):
                neighbors = _graph.neighborhood(vertices=_node, order=k_path_len, mode="out")
                subgraphs.append(_graph.induced_subgraph(neighbors))
    # Make dynamic
    dynamic_subgraphs = []
    for _graph in subgraphs:
        _dyn_graph = [_graph.subgraph_edges(_graph.es.select(subgraph=i),
                                            delete_vertices=False) for i in range(n_intervals)]        
        dynamic_subgraphs.append(_dyn_graph)
    # Create torch dataset
    dataset_list = []
    for _dynamic_graph in dynamic_subgraphs:
        _torch_dynamic_graph = []
        for _graph in _dynamic_graph:
            # Features first
            features = torch.vstack((
                torch.tensor(_graph.vs["score"], dtype=torch.float),
                torch.tensor(_graph.indegree(), dtype=torch.float),
                torch.tensor(_graph.outdegree(), dtype=torch.float)
            )).t()
            # Simplified edges set
            edge_index = torch.tensor(
                _graph.simplify(multiple=True, loops=False).get_edgelist(),
                dtype=torch.int64).t().contiguous()
            if edge_index.size(0) == 0 or features.size(0) < min_nodes:
                break 
            data = Data(x=features, edge_index=edge_index)
            assert data.validate(), "Incorrect graph formulation"
            _torch_dynamic_graph.append(data)
        if len(_torch_dynamic_graph) == len(_dynamic_graph):
            dataset_list.append(Batch.from_data_list(_torch_dynamic_graph))
    if len(dataset_list) > 0:
        dataset = Batch.from_data_list(dataset_list)
    else:
        dataset = None

    return dataset

# %% 
if __name__ == "__main__":
    ds = kpath_dyngraphs(
        dataset_file="./data/dataset_003/auth_ddcrp.txt",
        scores_file="./data/dataset_003/DP/phase_4_source_score.txt",
        max_t=24 * 60 * 60,
        interval=15 * 60,
        n_intervals=4,
        k_path_len=3)
