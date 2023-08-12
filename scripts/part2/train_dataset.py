"""Create training dataset for anomaly detection part 2"""
# %% Imports
import pandas as pd
import igraph as ig
from anomaly_detection.utils import switched_open

# %% Load dataset (time, source, destination, anomaly)
DATASET = "./data/dataset_003/auth_ddcrp.txt"
# Source scores
SCORES = "./data/dataset_003/DP/phase_4_source_score.txt"

# Parameters
MAX_T_TRAINING = 24 * 60 * 60
INTERVAL = 15 * 60
N_TIMES = 4
K_NEIGHBOURS = 3

# %% Source nodes
source_nodes = pd.read_table(SCORES, names=("score", "timewhen"), index_col=0,
                             dtype={"score": float, "timewhen": float})

# %% Create graphs
macro_graphs = []
current_links_list = []
subgraph_list = []
current_t0 = None
current_group = None
with switched_open(DATASET, "r") as file:
    for line in file:
        time, source, destination, _ = line.strip().split("\t")
        time = float(time)
        if time > MAX_T_TRAINING:
            break
        macro_group = time // (INTERVAL * N_TIMES)
        if current_group is None:
            current_group = macro_group
        if current_t0 is None:
            current_t0 = time
        if macro_group != current_group:
            graph = ig.Graph.TupleList(current_links_list, directed=True, weights=False)
            graph.vs["score"] = source_nodes.loc[graph.vs["name"]].score.tolist()
            graph.es["subgraph"] = subgraph_list
            macro_graphs.append(graph)
            current_group = macro_group
            current_links_list = []
            subgraph_list = []
            current_t0 = time
        if source in source_nodes.index:
            subgraph_list.append(int((time - current_t0) // INTERVAL))
            current_links_list.append((source, destination))

# %% Extract graphs
subgraphs = []
for _node in source_nodes.index:
    for _graph in macro_graphs:
        if _graph.vs.select(name=_node):
            neighbors = _graph.neighborhood(vertices=_node, order=K_NEIGHBOURS, mode="out")
            subgraphs.append(_graph.induced_subgraph(neighbors))

# %% Make dynamic
dynamic_subgraphs = []
for _graph in subgraphs:
    _dyn_graph = [_graph.subgraph_edges(_graph.es.select(subgraph=i),
                                        delete_vertices=False) for i in range(N_TIMES)]        
    dynamic_subgraphs.append(_dyn_graph)

# %%
