"""Script to simulate lateral movement in the network"""
from typing import Union, List, Optional
from collections import Counter
from warnings import warn
import random
import numpy as np
from numpy.random._generator import Generator
from igraph import Graph, set_random_number_generator
from processes.poisson import poisson_process
from data_generation.data_generation import create_and_save_dataset

class LateralMovementType():
    """Parametrize lateral movement"""
    _random_walk_params = ["rw_steps", "rw_num", "rw_reset"]
    _longest_path_params = ["n_attempts"]
    lm_type = None
    def __new__(cls, **kwargs):
        if set(kwargs) == set(cls._random_walk_params):
            cls.lm_type = "random_walk"
        elif set(kwargs) == set(cls._longest_path_params):
            cls.lm_type = "longest_path"
        else:
            raise KeyError("Unexpected set of parameters")
        for i, j in kwargs.items():
            setattr(cls, i, j)
        return cls
    
def generate_lateral_movement(
        edge_list: List[tuple],
        time_interval: List[float],
        rate: float,
        typology: LateralMovementType,
        target_type: Optional[str]="low-traffic",
        gen: Optional[Union[int, Generator]]=None,
        discretize_time: bool=False,
        file_name: Optional[str]=None):
    """Generate lateral movement data
    
    Output parameters
    :param dicretize_time: a posteriory, only in the file
    :param file_name: save to file if filename present"""
    t0, t1 = time_interval
    times = t0 + poisson_process(
        rate=rate,
        tmax=t1 - t0,
        gen=gen
    )
    graph = lateral_movement_newtork(edge_list=edge_list)
    endpoints = lateral_movement_endpoints(
        graph=graph,
        target_type=target_type,
        gen=gen)
    if typology.lm_type == "random_walk":
        warn("Need to deal with seed")
        source_seq, target_seq = random_walk_lateral_movement(
            graph=graph,
            endpoints=endpoints,
            rw_steps=typology.rw_steps,
            rw_num=typology.rw_num,
            rw_reset=typology.rw_reset,
            max_len=len(times),
            seed=gen
        )
    elif typology.lm_type == "longest_path":
        source_seq, target_seq = longets_path_stubborn_lateral_movement(
            graph=graph,
            endpoints=endpoints,
            n_attempts=typology.n_attempts
        )
    else:
        NotImplementedError("Unknown lateral movement")
    # Truncate times in case the target was reached earlier than maximum time budget
    if len(times) > len(source_seq):
        times = times[:len(source_seq)]
    return create_and_save_dataset(
        times=times,
        sources=source_seq,
        destinations=target_seq,
        anomaly=np.ones(shape=len(times)),
        discretize_time=discretize_time,
        file_name=file_name)

# Helper functions

def lateral_movement_newtork(
        edge_list: List[tuple]):
    """Creates an `igraph` to be used for the lateral movement strategy"""
    edge_counter = Counter(edge_list)
    edge_list_unique = list(edge_counter)
    edge_list_count = list(edge_counter.values())
    weighted_edge_list_unique = [
        (*_edge, _count) for _edge, _count in zip(edge_list_unique, edge_list_count)]
    return Graph.TupleList(weighted_edge_list_unique, directed=True, weights=True)
    
def _rename_output_sequence(graph: Graph, seq: list):
    """Rename output"""
    vertex_df = graph.get_vertex_dataframe()
    return [vertex_df.iloc[i]["name"] for i in seq]

def lateral_movement_endpoints(
        graph: Graph,
        target_type: str="low-traffic",
        gen: Optional[Union[int, Generator]]=None):
    """Given a graph calculates lateral movement source and target. Source is assumed
    to be a not-sensitive node, presumably easier to access in the first place.
    
    :param str target_type: defines if the target machine is interested by high activity
                            or is quite remote
    :return liable_vertex: lateral movement source
    :return target_vertex: lateral movement target
    """
    if gen is None or isinstance(gen, int):
        gen = np.random.default_rng(seed=gen)
    assert target_type in ["low-traffic", "high-traffic"], 'Unexpected target type!'
    try:
        node_importance = np.array(graph.pagerank(weights=graph.es["weight"]))
    except Exception as e:
        print(e)
        node_importance = np.array(graph.outdegree())
    # Pick a liability from least important nodes, a target depending on input
    separator = np.quantile(node_importance, 0.75)
    liable_vertex = gen.choice(np.where(node_importance < separator)[0])
    if target_type == "high-traffic":
        target_vertex = gen.choice(np.where(node_importance > separator)[0])
    else:
        target_vertex = liable_vertex
        # Make sure liable and target are not the same and are somehow connected
        while target_vertex == liable_vertex or \
            len(graph.get_shortest_path(liable_vertex, target_vertex, mode="all")) < 1:
            target_vertex = gen.choice(np.where(node_importance < separator)[0])
    return liable_vertex, target_vertex

def random_walk_lateral_movement(
        graph: Graph,
        endpoints: List[int],
        rw_steps: int=10,
        rw_num: int=10,
        rw_reset: bool=False,
        max_len: Optional[int]=1e8,
        seed: int=None
        ):
    """Creates a burst of random walks according to the graph until the target node
    is reached, or the maximum sequence length is reached.
    
    Parameters:
    :param graph: self-explanatory 
    :param endpoints: list containing source and target nodes
    :param rw_steps: length of random walks
    :param rw_num: dimension of each batch of random walks
    :param rw_reset: if True, each batch restarts from the initial liable node
    :param max_len: cap on numper of connections"""
    set_random_number_generator(random.seed(seed))
    liable_vertex = endpoints[0]
    target_vertex = endpoints[1]
    random_walks = np.ndarray(shape=(0, rw_steps + 1))
    rw_sources = np.repeat(liable_vertex, rw_num)
    random_walk_sources = []
    random_walk_targets = []
    while not np.any(random_walks == target_vertex) and len(random_walk_sources) < max_len:
        rw = np.array([graph.random_walk(start, steps=rw_steps, mode="all") for \
                       start in rw_sources])
        random_walks = np.vstack((random_walks, rw))
        random_walk_sources.extend(rw.T.flatten()[:-rw.shape[0]].tolist())
        random_walk_targets.extend(rw.T.flatten()[rw.shape[0]:].tolist())
        if not rw_reset:
            rw_sources = rw[:, -1]
    if len(random_walk_sources) > max_len:
        random_walk_sources = random_walk_sources[:max_len]
        random_walk_targets = random_walk_targets[:max_len]
    return _rename_output_sequence(graph, random_walk_sources), \
        _rename_output_sequence(graph, random_walk_targets)

def longets_path_stubborn_lateral_movement(
        graph: Graph,
        endpoints: List[int],
        n_attempts: int=10
        ):
    """Lateral movement following longest path, with repeated attempts"""
    liable_vertex = endpoints[0]
    target_vertex = endpoints[1]
    longest_path_sources = []
    longest_path_targets = []
    longest_path = graph.get_all_shortest_paths(
        liable_vertex, to=target_vertex, weights=[1 / i for i in graph.es["weight"]],
        mode="all")
    longest_path_repeated = np.repeat(longest_path, n_attempts)
    longest_path_sources.extend(longest_path_repeated.T.flatten()[:-n_attempts].tolist())
    longest_path_targets.extend(longest_path_repeated.T.flatten()[n_attempts:].tolist())
    return _rename_output_sequence(graph, longest_path_sources), \
        _rename_output_sequence(graph, longest_path_targets)
