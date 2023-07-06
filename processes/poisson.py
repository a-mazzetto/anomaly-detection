"""Generation of homogeneous and inhomogeneous Poisson Processes"""
import numpy as np
from numpy.random._generator import Generator
from typing import List, Optional, Union

def _check_length_or_max_alternative(_length, _max):
    """One nor the other must be spcified"""
    if not np.logical_xor(_length is None, _max is None):
        raise ValueError('Must specify length XOR max')

def poisson_process(
        rate: Optional[float]=1.0,
        length: Optional[int]=None,
        tmax: Optional[float]=None,
        gen: Optional[Union[int, Generator]]=None) -> np.ndarray:
    """Function to draw samples from a Poisson Point Process.
    
    :param float rate: intensity parameter
    :param int length: stopping rule based on number of samples (alternative to max)
    :param float tmax: stopping rule based on maximum distance to cover (alternative to length)
    :param gen: integer or generator (optional)
    :return np.ndarray samples: samples"""
    if gen is None or isinstance(gen, int):
        gen = np.random.default_rng(seed=gen)
    if rate < 0:
        raise ValueError('Rate parameter must be positive')
    _ = _check_length_or_max_alternative(length, tmax)
    if length is not None:
        samples = np.cumsum(
            np.append(0, gen.exponential(scale=1 / rate, size=length)))
    else:
        samples = np.array((0,))
        while samples[-1] < tmax:
            samples = np.append(
                samples,
                samples[-1] + gen.exponential(scale=1 / rate))

    return samples

def n_independent_poisson_processes(
    rates: Union[List[float], np.ndarray],
    length: Optional[int]=None,
    tmax: Optional[float]=None,
    gen: Optional[Union[int, Generator]]=None) -> np.ndarray:
    """Function to draw samples from independent Poisson Point Processes
    simultaneously.
    
    :param List[float] rates: intensity parameters for each process
    :param int length: stopping rule based on number of samples (alternative to max)
    :param float max: stopping rule based on maximum distance to cover (alternative to length)
    :return np.ndarray samples: samples"""
    if gen is None or isinstance(gen, int):
        gen = np.random.default_rng(seed=gen)
    rates = np.array(rates) if isinstance(rates, list) else rates
    if np.any(rates < 0):
        raise ValueError('Rate parameter must be positive')
    # Calculate the full rate parameter
    rate = sum(rates)
    partition_probs = rates / rate
    # Calculate the total Poisson process
    samples = poisson_process(rate, length, tmax, gen=gen)
    # Partition a posteriori
    partition = gen.choice(a=np.arange(len(partition_probs)), size=len(samples), p=partition_probs)

    return samples, partition

def partitioned_poisson_process(
        rate: float,
        partition_probs: Union[List[float], np.ndarray],
        length: Optional[int]=None,
        tmax: Optional[float]=None,
        gen: Optional[Union[int, Generator]]=None) -> np.ndarray:
    """Function to draw samples from a Poisson Point Process.
    
    :param List[float] rate: intensity parameters of the global process
    :param List[float] partition_probs: partition probabilities
    :param int length: stopping rule based on number of samples (alternative to max)
    :param float max: stopping rule based on maximum distance to cover (alternative to length)
    :return np.ndarray samples: samples"""
    partition_probs = np.array(partition_probs) if isinstance(partition_probs, list) else partition_probs
    if not np.isclose(sum(partition_probs), 1.):
        raise ValueError('Partition probabilities must sum to one!')
    rates = rate * partition_probs
    samples, partition = n_independent_poisson_processes(
        rates, length, tmax, gen=gen
    )
    return samples, partition

def inhomogeneous_poisson_process_sinusoidal(
        rate0: float,
        rate1: float,
        period: float,
        length: Optional[int]=None,
        tmax: Optional[float]=None,
        gen: Optional[Union[int, Generator]]=None):
    """Inhomogeneous Poisson Point Process with rate defined as
    $\lambda = \lambda_0 + \lambda_1 sin((2 \pi / T) t)$ and
    simulated using a thinning algorithm
    
    :param float rate0: carrier amplitude
    :param float rate1: periodic amplitude
    :param float period: periodicty
    :param int length: stopping rule based on number of samples (alternative to max)
    :param float tmax: stopping rule based on maximum distance to cover (alternative to length)
    :return np.ndarray samples: samples"""
    if gen is None or isinstance(gen, int):
        gen = np.random.default_rng(seed=gen)
    # define rate with sinusoidal component
    def _rate(_time):
        """_time is absolute time"""
        return rate0 + rate1 * np.sin(2 * np.pi / period * _time)
    max_rate = rate0 + rate1
    if rate1 >= rate0 or rate0 < 0 or rate1 < 0:
        raise ValueError('Rates must be positive and the carrier must be greater')
    # Generate the full process with max_rate
    sequence = poisson_process(max_rate, length, tmax, gen)
    # Challenge each time-step
    challenge = gen.uniform(size=len(sequence)) < _rate(sequence) / max_rate
    return sequence[challenge]
