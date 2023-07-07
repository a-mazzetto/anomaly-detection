"""Functions related to Dirichlet Process"""
from typing import Callable, Union, List, Optional
from warnings import warn
import numpy as np
from numpy.random._generator import Generator
from scipy.stats import geom

def sample_dirichlet_process_geometric_base(
        intensity: float,
        theta: float,
        n: int,
        gen: Generator):
    """Sample from Dirichlet Process given base measure p0
    Geom(th) probabilities p(X=1),..,p(X=m), then p(X>m)
    
    :param intensity: DP intensity parameter (the higher, the closer to base measure)
    :param theta: geometric distribution parameter
    :param n: number of points
    :param gen: random numbers generator"""
    p = [geom(theta).pmf(k) for k in range(1, n)] + [geom(theta).sf(n - 1)]
    assert np.isclose(sum(p), 1.), 'Probability must sum to 1!'
    return gen.dirichlet(intensity * np.array(p))

def check_exp_ddcrp_parameters(intensity, decay):
    assert intensity >= 0, "DDDP intensity must be non-negative"
    assert decay > 0, "DDDP decay must be positive"

class DDChineseRestaurantProcessEnv():
    """Environment for distance dependent CRP with exponential decay"""
    def __init__(
            self,
            labels: Union[List[str], int],
            intensity: float=0.0,
            decay: float=0.0,
            seed: Optional[int]=None):
        if seed is None or isinstance(seed, int):
            self.gen = np.random.default_rng(seed=seed)
        else:
            self.gen = seed
        check_exp_ddcrp_parameters(intensity, decay)
        self.alpha = intensity
        self.decay = decay

        if isinstance(labels, int):
            self.masses = np.arange(labels)
            self.name_conversion = None
        else:
            self.masses = np.arange(len(labels))
            self.name_conversion = dict(zip(self.masses, labels))

        self.reset()

    def reset(self):
        "Reset counter and running sum of exponentials"
        self.n = 0
        self.counter = {}
        # We also have a time state
        self.state = -1e-8

    def _update(self, sample, time):
        "Update counts running sum of exponential"
        self.n += 1
        if sample in self.counter:
            self.counter[sample] += np.exp(time / self.decay)
        else:
            self.counter[sample] = np.exp(time / self.decay)
        self.state = time

    def sample(self, time):
        """Get a sample"""
        assert time > self.state, "Time must be monotonically increasing"
        sum_exp_i = np.array(list(self.counter.values()))

        new_key = -1
        if self.n == 0:
            choice = new_key
        else:
            p_new_sample = self.alpha / (self.alpha + self.n)
            pi = sum_exp_i * (self.n / np.sum(sum_exp_i))
            p_old_sample = pi / (self.alpha + self.n)
            choice = self.gen.choice([new_key] + list(self.counter),
                                     p=np.append(p_new_sample, p_old_sample))

        if choice == new_key:
            # Here we have a customer sitting with itself, must be a new cluster
            warn("Stll to decide what to do with DDDP base measure")
            choice = self.gen.choice(self.masses)
        self._update(choice, time)

        if self.name_conversion is not None:
            return self.name_conversion[choice]
        else:
            return choice

def generate_exp_ddcrp(
        intensity: Optional[float]=0.0,
        decay: Optional[float]=1.0,
        times: Optional[np.ndarray]=np.array((1,)),
        labels: Union[List[str], int]=1,
        seed: Optional[Union[int, Generator]]=None):
    """Function to generate samples following a distribution sampled from
    a distance-dependent DP. The function assumes a `discrete uniform base measure`
    
    :param float intensity: CRP intensity parameter
    :param float decay: distance decay factor
    :param int times: time sequence
    :param labels: labels or number of labels
    :type labels: int or list of strings
    """
    ddcrp = DDChineseRestaurantProcessEnv(labels=labels, intensity=intensity, decay=decay, seed=seed)
    sequence = [ddcrp.sample(_t) for _t in times]
    return sequence
