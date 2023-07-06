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
