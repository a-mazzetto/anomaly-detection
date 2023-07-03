"""Script containing all functions necessary for data generation"""
import numpy as np
from typing import List, Optional, Union
from processes.pitman_yor import PitmanYorEnv

def generate_pitman_yor(
        discount: Optional[float]=0.0,
        intensity: Optional[float]=0.0,
        length: Optional[int]=1,
        labels: Union[List[str], int]=1,
        seed: Optional[int]=None):
    """Function to generate samples following a distribution sampled from
    a Pitman-Yor process. The function assumes a `discrete uniform base measure`
    
    :param float discount: discount parameter
    :param float intensity: intensity parameter
    :param int length: length of the sequence to be generated
    :param labels: labels or number of labels
    :type labels: int or list of strings
    """
    pitman_yor = PitmanYorEnv(labels=labels, intensity=intensity, discount=discount, seed=seed)
    return [pitman_yor.sample() for _ in range(length)]

def generate_poisson_process(
        rate: Optional[float]=1.0,
        length: Optional[int]=None,
        max: Optional[float]=None,
        seed: Optional[int]=None) -> np.ndarray:
    """Function to draw samples from a Poisson Point Process.
    
    :param float intensity: intensity parameter
    :param int length: stopping rule based on number of samples (alternative to max)
    :param float max: stopping rule based on maximum distance to cover (alternative to length)
    :return np.ndarray samples: samples"""
    gen = np.random.default_rng(seed=seed)
    if rate < 0:
        raise ValueError('Intensity parameter must be positive')
    if not np.logical_xor(length is None, max is None):
        raise ValueError('Must specify length XOR max')
    if length is not None:
        samples = np.cumsum(
            np.append(0, gen.exponential(scale=1 / rate, size=length)))
    else:
        samples = np.array((0,))
        while samples[-1] < max:
            samples = np.append(
                samples,
                samples[-1] + gen.exponential(scale=1 / rate))

    return samples
