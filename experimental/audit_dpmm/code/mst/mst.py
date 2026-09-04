# A generative model training algorithm based on
# "Winning the NIST Contest: A scalable and general approach to differentially private synthetic data"
# by Ryan McKenna, Gerome Miklau, Daniel Sheldon
# Adapted from: https://github.com/ryan112358/private-pgm/blob/1da21c8b38149b05f1385b8e54116568b700b4fa/mechanisms/mst.py
# and
# Adapted from: https://github.com/sassoftware/dpmm/blob/752fd57480ec593a3b2b5950fd445e98cdedd7e3/src/dpmm/models/mst.py


import numpy as np
from logging import getLogger
from typing import Tuple, Optional
from numpy.random import RandomState

from dpmm.models.base.mbi import Dataset, Domain
from dpmm.models.base.mechanisms import cdp_rho
from dpmm.models.base.memory import model_size
from dpmm.models.base.mechanisms import Mechanism

from mst import mu_from_eps_delta


"""
This is a generalization of the winning mechanism from the
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.
"""


logger = getLogger("dpmm")


class MST(Mechanism):
    """
    Maximum Spanning Tree (MST) mechanism is a differentially private generative model relying
    on selecting an optimal set of marginals to approximate the joint distribution of the data.
    It uses the exponential mechanism to select higher-order marginals based on their weights.
    The marginals are measured using the Laplace mechanism.
    The measured marginals are then used to estimate a maximum spanning tree which will be able to generate data.

    Ref: https://arxiv.org/pdf/2108.04978

    :param epsilon: Privacy budget.
    :type epsilon: float, optional
    :param delta: Privacy parameter.
    :type delta: float, optional
    :param n_iters: Number of iterations for inference.
    :type n_iters: int
    :param compress: Whether to compress the data.
    :type compress: bool
    :param domain: The domain of the data.
    :type domain: Domain, optional
    :param prng: Random state for reproducibility.
    :type prng: RandomState, optional
    :param max_model_size: Maximum model size in MB.
    :type max_model_size: int, optional
    :param structural_zeros: Structural zeros in the data.
    :type structural_zeros: dict, optional
    :param n_jobs: Number of parallel jobs.
    :type n_jobs: int
    """

    def __init__(
        self,
        domain: Domain,
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        n_iters: int = 5000,
        compress: bool = False,
        GDP: bool = False,
        prng: Optional[RandomState] = None,
        max_model_size: Optional[int] = None,
        structural_zeros: Optional[dict] = None,
        n_jobs: int = -1,
    ):
        super().__init__(
            epsilon=epsilon,
            delta=delta,
            prng=prng,
            max_model_size=max_model_size,
            compress=compress,
            domain=domain,
            structural_zeros=structural_zeros,
            n_jobs=n_jobs,
        )

        if GDP:
            # HARDCODED -- 2 (convert ADP directly to GDP)
            self.rho = None
            mu = mu_from_eps_delta(self.epsilon, self.delta)
            self.sigma = 1 / mu
        else:
            self.rho = cdp_rho(self.epsilon, self.delta)
            # HARDCODED -- 1 (use all DP budget on 1-way marginals measurement)
            self.sigma = np.sqrt(1 / (2 * self.rho))

        self.n_iters = n_iters

    def _fit(self, data: Dataset, public: bool = False) -> Tuple[Dataset, list]:
        """
        Fit the MST mechanism to the data.

        :param data: The dataset.
        :type data: Dataset
        :param public: Whether the data is public. Defaults to False.
        :type public: bool, optional
        :return: The dataset and measurement log.
        :rtype: Tuple[Dataset, list]
        """
        # select all 1-way marginals
        self.cliques = cliques_1 = [(col,) for col in data.domain]

        log1 = self.measure(data, cliques=cliques_1, public=public)
        # compress domain of all 1-way marginals
        if self.compress:
            log1 = self.compressor.fit(log1)
            data = self.compressor.transform(data)

        self.model_size = model_size(data, cliques_1)
        # HARDCODED - 3 (only select all 1-way marginals)

        return data, log1
