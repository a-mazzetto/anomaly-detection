"""Script containing all functions necessary for data generation"""
# %%
from typing import List, Optional, Union, Callable
from copy import deepcopy
import numpy as np
from processes.poisson import inhomogeneous_poisson_process_sinusoidal
from processes.pitman_yor import generate_pitman_yor
from processes.dirichlet import generate_exp_ddcrp

def generate_dataset(
    max_time: float,
    period: float,
    n_nodes: int,
    destination_intensity: float,
    destination_discount: float,
    source_intensities: Union[float, Callable],
    source_discounts: Union[float, Callable],
    node_names: Optional[List[str]]=None,
    seed: Optional[int]=None,
    discretize_time: bool=False,
    file_name: Optional[str]=None):
    """Function to generate the dataset.

    Parameters for the Poisson Process. It is an inhomogeneous process
    with choice of rates to match the real dataset.
    :param max_time: maximum time
    :param period: sinusoidal periodicity

    Parameters for the PY/DP
    :param destination_intensity: constant
    :param destination_discount: constant
    :param source_intensities: constant for all or Callable (*)
    :param source_discounts: constant for all or Callable (*)

    (*) Must be a function that takes generator and sample size as inputs
    
    Output parameters
    :param dicretize_time: a posteriory, only in the file
    :param file_name: save to file if filename present
    """
    gen = np.random.default_rng(seed=seed)
    # generate time sequence
    times = generate_time_sequence(period=period, max_time=max_time, n_nodes=n_nodes, gen=gen)
    # Destinations PY/DP process
    destinations = generate_destination_sequence(discount=destination_discount,
        intensity=destination_intensity, node_names=node_names, times=times, gen=gen)
    # Finally, generate source conditioned upon destinations
    sources, source_intensity_vector, source_discount_vector = generate_source_sequence(
        discounts=source_discounts, intensities=source_intensities,
        node_names=node_names, destinations=destinations, gen=gen)
    # Create dataset
    dataset = create_and_save_dataset(
        times=times, destinations=destinations, sources=sources, anomaly=np.zeros_like(times),
        discretize_time=discretize_time, file_name=file_name)
    return dataset, source_intensity_vector, source_discount_vector

def generate_ddcrp_dataset(
    max_time: float,
    period: float,
    n_nodes: int,
    destination_intensity: float,
    destination_discount: float,
    source_intensities: Union[float, Callable],
    source_decay: float,
    node_names: Optional[List[str]]=None,
    seed: Optional[int]=None,
    discretize_time: bool=False,
    file_name: Optional[str]=None):
    """Function to generate the dataset.

    Parameters for the Poisson Process. It is an inhomogeneous process
    with choice of rates to match the real dataset.
    :param max_time: maximum time
    :param period: sinusoidal periodicity

    Parameters for the destination PY/DP process
    :param destination_intensity: constant
    :param destination_discount: constant

    Parameters for the distance-dependent CRP
    :param source_intensities: constant for all or Callable (*)
    :param source_decay: exponential distance decay parameter

    (*) Must be a function that takes generator and sample size as inputs
    
    Output parameters
    :param dicretize_time: a posteriory, only in the file
    :param file_name: save to file if filename present
    """
    gen = np.random.default_rng(seed=seed)
    # generate time sequence
    times = generate_time_sequence(period=period, max_time=max_time, n_nodes=n_nodes, gen=gen)
    # Destinations PY/DP process
    destinations = generate_destination_sequence(discount=destination_discount,
        intensity=destination_intensity, node_names=node_names, times=times, gen=gen)
    # Finally, generate source conditioned upon destinations, with time dependency
    sources, source_intensity_vector = generate_ddcrp_source_sequence(
        intensities=source_intensities, decay=source_decay, node_names=node_names,
        destinations=destinations, times=times, gen=gen)
    # Create dataset
    dataset = create_and_save_dataset(
        times=times, destinations=destinations, sources=sources, anomaly=np.zeros_like(times),
        discretize_time=discretize_time, file_name=file_name)
    return dataset, source_intensity_vector

# Helper functions

def generate_time_sequence(period, max_time, n_nodes, gen):
    """Helper function to generate time sequence"""
    # Inhomogeneous Poisson Process with rates to emulate LANL dataset
    # From LANL dataset: 0.8-1.4M/4h connections with 12000 nodes
    lambda_min_max = np.array((0.8, 1.4)) * 1e6 / 12000 / (4 * 60 * 60) * n_nodes
    lambda_0 = np.mean(lambda_min_max)
    lambda_1 = np.diff(lambda_min_max)[-1] / 2
    times = inhomogeneous_poisson_process_sinusoidal(
        lambda_0, lambda_1, period, max_time, gen=gen
        )
    return times

def generate_destination_sequence(discount, intensity, node_names, times, gen):
    """Helper function to generate destination sequency"""
    destinations = generate_pitman_yor(
        discount=discount,
        intensity=intensity,
        length=len(times),
        labels=node_names,
        seed=gen
    )
    return destinations

def _parse_source_pitman_yor_parameters(discounts, intensities, unique_destinations, gen):
    """Helper function to parse source parameters"""
    n_unique_destinations = len(unique_destinations)
    if isinstance(intensities, (float, int)):
        source_intensity_vector = intensities * np.ones(n_unique_destinations)
    else:
        source_intensity_vector = intensities(gen=gen, size=n_unique_destinations)
    if isinstance(discounts, (float, int)):
        source_discount_vector = discounts * np.ones(n_unique_destinations)
    else:
        source_discount_vector = discounts(gen=gen, size=n_unique_destinations)
    return source_intensity_vector, source_discount_vector

def _parse_source_ddcrp_parameters(intensities, unique_destinations, gen):
    """Helper function to parse source parameters"""
    n_unique_destinations = len(unique_destinations)
    if isinstance(intensities, (float, int)):
        source_intensity_vector = intensities * np.ones(n_unique_destinations)
    else:
        source_intensity_vector = intensities(gen=gen, size=n_unique_destinations)
    return source_intensity_vector

def generate_source_sequence(discounts, intensities, node_names, destinations, gen):
    """Helper function to calculate sources conditional upon destinations.
    Recall that `discounts` and `intensities` can be Callable"""
    unique_destinations = np.unique(destinations)
    source_intensity_vector, source_discount_vector = _parse_source_pitman_yor_parameters(
        discounts=discounts,
        intensities=intensities,
        unique_destinations=unique_destinations,
        gen=gen
    )
    # Generate processes
    link_dict = {}
    for _dest, _int, _disc in zip(unique_destinations, source_intensity_vector, source_discount_vector):
        _n = destinations.count(_dest)
        _sources = generate_pitman_yor(
            discount=_disc,
            intensity=_int,
            length=_n,
            labels=node_names,
            seed=gen
        )
        link_dict[_dest] = _sources
    # Return list of sources preserving chronological order
    sequence = [link_dict[_dest].pop(0) for _dest in destinations]
    assert all(len(i) == 0 for i in link_dict.values()), 'All sources should be consumed by now!'
    return sequence, source_intensity_vector, source_discount_vector

def generate_ddcrp_source_sequence(intensities, decay, node_names, destinations, times, gen):
    """Helper function to calculate sources conditional upon destinations.
    Recall that `discounts` and `intensities` can be Callable"""
    unique_destinations = np.unique(destinations)
    source_intensity_vector = _parse_source_ddcrp_parameters(
        intensities=intensities,
        unique_destinations=unique_destinations,
        gen=gen
    )
    # Generate processes
    link_dict = {}
    for _dest, _int in zip(unique_destinations, source_intensity_vector):
        _times = np.array([_t for _t, _d in zip(times, destinations) if _d == _dest])
        _times = _times - _times[0]
        _sources = generate_exp_ddcrp(
            intensity=_int,
            decay=decay,
            times=_times,
            labels=node_names,
            seed=gen
        )
        link_dict[_dest] = _sources
    # Return list of sources preserving chronological order
    sequence = [link_dict[_dest].pop(0) for _dest in destinations]
    assert all(len(i) == 0 for i in link_dict.values()), 'All sources should be consumed by now!'
    return sequence, source_intensity_vector

def create_and_save_dataset(times, destinations, sources, anomaly, discretize_time, file_name):
    """Helper function to create and save dataset"""
    if discretize_time:
        times = np.floor(times).astype(int)
    dataset = list(zip(times, sources, destinations, anomaly.astype(int)))
    # Save dataset if required
    save_dataset(dataset, file_name=file_name)
    return dataset

def save_dataset(dataset, file_name):
    """save dataset to file"""
    if file_name is not None:
        with open(file_name, 'w', encoding='utf-8') as file:
            for line in dataset:
                file.write('\t'.join(str(s) for s in line) + '\n')
