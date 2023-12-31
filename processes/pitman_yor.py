"""Script containing Pitman-Yor process utilities"""
from typing import List, Union, Optional
import numpy as np
from numpy.random._generator import Generator
from processes.dirichlet import sample_dirichlet_process_geometric_base

def check_pitman_yor_params(discount: float=0.0, intensity: float=0.0):
    """Chack validity of Pitman-Yor parameters"""
    if discount < 0 or discount >= 1:
        raise ValueError('Discount parameter must be in [0, 1)')
    if intensity < -discount:
        raise ValueError('Intensity parameter must be greater than minus discount')

class PitmanYorEnv():
    """Pitman-Yor Environment"""
    def __init__(
            self,
            labels: Union[List[str], int],
            intensity: float=0.0,
            discount: float=0.0,
            seed: Optional[int]=None):
        if seed is None or isinstance(seed, int):
            self.gen = np.random.default_rng(seed=seed)
        else:
            self.gen = seed
        check_pitman_yor_params(discount=discount, intensity=intensity)
        self.a = intensity
        self.d = discount

        if isinstance(labels, int):
            self.masses = np.arange(labels)
            self.name_conversion = None
        else:
            self.masses = np.arange(len(labels))
            self.name_conversion = dict(zip(self.masses, labels))

        self.reset()

    def reset(self):
        "Reset counter"
        self.n = 0
        self.counter = {}

    def _update(self, sample):
        "Update counts and total count"
        self.n += 1
        if sample in self.counter:
            self.counter[sample] += 1
        else:
            self.counter[sample] = 1

    def _anomalous_base_measure(self):
        raise NotImplementedError()

    def sample(self, is_anomalous=False):
        """Get a sample"""
        ni = list(self.counter.values())
        n_unique = len(self.counter)

        new_key = -1
        if self.n == 0 and self.a == 0:
            choice = new_key
        else:
            p_new_sample = [(self.a + self.d * n_unique)/(self.a + self.n)]
            p_old_sample = [(i - self.d)/(self.a + self.n) for i in ni]
            choice = self.gen.choice([new_key] + list(self.counter), p=p_new_sample + p_old_sample)

        if choice == new_key:
            # Note: with discrete base measure the new sample could be in the counter already
            if is_anomalous:
                choice = self._anomalous_base_measure()
            else:
                choice = self.gen.choice(self.masses)
        self._update(choice)

        if self.name_conversion is not None:
            return self.name_conversion[choice]
        else:
            return choice

def generate_pitman_yor(
        discount: Optional[float]=0.0,
        intensity: Optional[float]=0.0,
        length: Optional[int]=1,
        labels: Union[List[str], int]=1,
        seed: Optional[Union[int, Generator]]=None):
    """Function to generate samples following a distribution sampled from
    a Pitman-Yor process. The function assumes a `discrete uniform base measure`
    
    :param float discount: discount parameter
    :param float intensity: intensity parameter
    :param int length: length of the sequence to be generated
    :param labels: labels or number of labels
    :type labels: int or list of strings
    """
    pitman_yor = PitmanYorEnv(labels=labels, intensity=intensity, discount=discount, seed=seed)
    sequence = [pitman_yor.sample() for _ in range(length)]
    return sequence

class PitmanYorEnvWithAnomaly(PitmanYorEnv):
    """Introduction of anomaly"""
    def __init__(
            self,
            labels: Union[List[str], int],
            anomaly_base_measure: List[np.ndarray],
            intensity: float=0.0,
            discount: float=0.0,
            seed: Optional[int]=None):
        super().__init__(labels, intensity, discount, seed)
        # Check validity of anomaly base measure
        if not np.isclose(sum(anomaly_base_measure[1]), 1.):
            raise ValueError('Anomalous base measure probability must sum to one')
        try:
            if self.name_conversion is None:
                assert sorted(self.masses) == sorted(anomaly_base_measure[0])
            else:
                assert sorted(anomaly_base_measure[0]) == sorted(labels)
        except Exception as e:
            print(e)
            raise ValueError('Likely inconsistent anomalous base measure')
        self.anomaly_base_measure = anomaly_base_measure
    
    def _anomalous_base_measure(self):
        return self.gen.choice(
            self.anomaly_base_measure[0],
            p=self.anomaly_base_measure[1])

def generate_pitman_yor_with_anomaly(
        discount: Optional[float]=0.0,
        intensity: Optional[float]=0.0,
        length: Optional[int]=1,
        labels: Union[List[str], int]=1,
        anomaly: Optional[np.ndarray]=None,
        anomaly_base_measure: Optional[List[np.ndarray]]=None,
        seed: Optional[Union[int, Generator]]=None):
    """Function to generate samples following a distribution sampled from
    a Pitman-Yor process. The function assumes a `discrete uniform base measure`
    
    :param float discount: discount parameter
    :param float intensity: intensity parameter
    :param int length: length of the sequence to be generated
    :param labels: labels or number of labels
    :type labels: int or list of strings
    :param np.ndarray anomaly: boolean vector of
    :param np.ndarray anomaly: vector of length `labels` as alternative
    """
    assert len(anomaly) == length, "Anomaly boolean vector of unexpected size"
    pitman_yor = PitmanYorEnvWithAnomaly(labels=labels, intensity=intensity, discount=discount,
                                         anomaly_base_measure=anomaly_base_measure, seed=seed)
    sequence = [pitman_yor.sample(is_anomalous=b) for b in anomaly]
    return sequence
