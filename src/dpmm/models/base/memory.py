from typing import Tuple, List
import numpy as np

from dpmm.models.base.mbi.dataset import Dataset


def meas_size(shape: Tuple):
    return np.prod(shape) / (8 * (1028**2))


def clique_size(data: Dataset, clique: Tuple):
    return meas_size(data.project(clique).domain.shape)


def model_size(data: Dataset, cliques: List[Tuple]):
    return sum([clique_size(data, clique) for clique in cliques])
