"""Script containing all functions necessary for data generation"""
# %%
from typing import List, Optional, Union, Callable
from copy import deepcopy
import numpy as np
from processes.poisson import inhomogeneous_poisson_process_sinusoidal
from processes.pitman_yor import generate_pitman_yor

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
    dicretize_time: bool=False,
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
    # Inhomogeneous Poisson Process with rates to emulate LANL dataset
    # From LANL dataset: 0.8-1.4M/4h connections with 12000 nodes
    lambda_min_max = np.array((0.8, 1.4)) * 1e6 / 12000 / (4 * 60 * 60) * n_nodes
    lambda_0 = np.mean(lambda_min_max)
    lambda_1 = np.diff(lambda_min_max)[-1] / 2
    times = inhomogeneous_poisson_process_sinusoidal(
        lambda_0, lambda_1, period, max_time, gen=gen
        )
    # Destinations PY/DP process
    destinations = generate_pitman_yor(
        discount=destination_discount,
        intensity=destination_intensity,
        length=len(times),
        labels=node_names,
        seed=gen
    )
    # Finally, generate source conditioned upon destinations
    unique_destinations = np.unique(destinations)
    n_unique_destinations = len(unique_destinations)
    if isinstance(source_intensities, (float, int)):
        source_intensity_vector = source_intensities * np.ones(len(destinations))
    else:
        source_intensity_vector = source_intensities(
            gen=gen,
            size=n_unique_destinations)
    if isinstance(source_discounts, (float, int)):
        source_discount_vector = source_discounts * np.ones(len(destinations))
    else:
        source_discount_vector = source_discounts(
            gen=gen,
            size=n_unique_destinations)
    link_dict = {}
    for _dest, _int, _disc in zip(unique_destinations,
                               source_intensity_vector,
                               source_discount_vector):
        _n = destinations.count(_dest)
        _sources = generate_pitman_yor(
            discount=_disc,
            intensity=_int,
            length=_n,
            labels=node_names,
            seed=gen
        )
        link_dict[_dest] = _sources
    # Create link list preserving generation order
    dataset = []
    for _time, _dest in zip(times, destinations):
        dataset.append((
            _time if not dicretize_time else np.floor(_time).astype(int),
            link_dict[_dest].pop(0),
            _dest))
    # Save dataset if required
    if file_name is not None:
        with open(file_name, 'w', encoding='utf-8') as file:
            for line in dataset:
                file.write(' '.join(str(s) for s in line) + '\n')
    return dataset, source_intensity_vector, source_discount_vector
