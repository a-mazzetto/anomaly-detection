"""Functions related to Dirichlet Process"""
from typing import Callable
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
