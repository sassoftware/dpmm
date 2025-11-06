# A generative model training algorithm based on
# "Winning the NIST Contest: A scalable and general approach to differentially private synthetic data"
# by Ryan McKenna, Gerome Miklau, Daniel Sheldon
# Adapted from: https://github.com/ryan112358/private-pgm/blob/1da21c8b38149b05f1385b8e54116568b700b4fa/mechanisms/mst.py


# import argparse
import itertools
from logging import getLogger
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Self, Tuple, Optional

import joblib
import networkx as nx
import numpy as np
from disjoint_set import DisjointSet
from numpy.random import RandomState
from scipy.special import logsumexp

from dpmm.models.base.graphical import GraphicalGenerativeModel
from dpmm.models.base.mbi import Dataset, Domain, FactoredInference, GraphicalModel
from dpmm.models.base.mechanisms import cdp_rho
from dpmm.models.base.memory import clique_size, model_size
from dpmm.utils import to_path
from dpmm.models.base.mechanisms import Mechanism

"""
This is a generalization of the winning mechanism from the
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.
"""


logger = getLogger("dpmm")


def compute_weight(
    est: FactoredInference, data: Dataset, clique: Tuple[str, str]
) -> Tuple[str, str, float, int]:
    """
    Compute the weight of a clique based on the difference between the estimated and actual data vectors.

    :param est: The inference engine used to estimate the data.
    :type est: FactoredInference
    :param data: The dataset.
    :type data: Dataset
    :param clique: A tuple representing the clique (pair of attributes).
    :type clique: Tuple[str, str]
    :return: A tuple containing the clique attributes, weight, and model size.
    :rtype: Tuple[str, str, float, int]
    """
    a, b = clique
    model_size = clique_size(data, (a, b))
    xhat = est.project([a, b]).datavector()
    x = data.project([a, b]).datavector()
    weight = np.linalg.norm(x - xhat, 1)
    return a, b, weight, model_size


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
        epsilon: Optional[float] = None,
        delta: Optional[float] = None,
        n_iters: int = 10000,
        compress: bool = True,
        domain: Optional[Domain] = None,
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

        self.rho = cdp_rho(self.epsilon, self.delta)
        self.sigma = np.sqrt(3 / (2 * self.rho))
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
        # select higher marginals
        if len(data.domain.attrs) > 1:
            cliques_2 = self.select(data, self.rho / 3.0, log1)
            # measure higher marginals
            log2 = self.measure(data, cliques=cliques_2, flatten=True)
            self.cliques += cliques_2
            # TODO document
        else:
            log2 = []

        return data, log1 + log2

    def select(  # noqa: C901
        self,
        data: Dataset,
        rho: float,
        measurement_log: list,
        cliques: list = [],
        public: bool = False,
    ) -> list:
        """
        Select higher-order marginals using the exponential mechanism.

        :param data: The dataset.
        :type data: Dataset
        :param rho: Privacy budget for selection.
        :type rho: float
        :param measurement_log: Log of measurements.
        :type measurement_log: list
        :param cliques: Existing cliques. Defaults to an empty list.
        :type cliques: list, optional
        :param public: Whether the data is public. Defaults to False.
        :type public: bool, optional
        :return: A list of selected cliques.
        :rtype: list
        """
        engine = FactoredInference(data.domain, iters=2500)
        est = engine.estimate(measurement_log)

        weights = {}
        candidates = list(itertools.combinations(data.domain.attrs, 2))
        if self.n_jobs > 1:
            for a, b, weight, m_size in Pool(self.n_jobs).starmap(
                compute_weight,
                zip(itertools.cycle([est]), itertools.cycle([data]), candidates),
            ):
                weights[a, b] = weight
                self.model_size += m_size
                if self.max_model_size is not None:
                    if self.model_size > self.max_model_size:
                        break
        else:
            for compute_args in zip(
                itertools.cycle([est]), itertools.cycle([data]), candidates
            ):
                a, b, weight, m_size = compute_weight(*compute_args)
                weights[a, b] = weight
                self.model_size += m_size
                if self.max_model_size is not None:
                    if self.model_size > self.max_model_size:
                        break

        graph = nx.Graph()
        graph.add_nodes_from(data.domain.attrs)
        ds = DisjointSet()

        for e in cliques:
            graph.add_edge(*e)
            ds.union(*e)

        r = len(list(nx.connected_components(graph)))
        epsilon = np.sqrt(8 * rho / (r - 1))
        for i in range(r - 1):
            candidates = [e for e in candidates if not ds.connected(*e)]
            wgts = np.array([weights[e] for e in candidates])
            if public:
                idx = np.argmax(wgts)
            else:
                idx = self.exponential_mechanism(wgts, epsilon, sensitivity=1.0)

            e = candidates[idx]
            graph.add_edge(*e)
            ds.union(*e)

        return list(graph.edges)

    def exponential_mechanism(
        self, q: np.ndarray, eps: float, sensitivity: float, monotonic: bool = False
    ) -> int:
        """
        Apply the exponential mechanism to select a candidate.

        :param q: Quality scores for candidates.
        :type q: np.ndarray
        :param eps: Privacy budget.
        :type eps: float
        :param sensitivity: Sensitivity of the scores.
        :type sensitivity: float
        :param monotonic: Whether the scores are monotonic. Defaults to False.
        :type monotonic: bool, optional
        :return: The index of the selected candidate.
        :rtype: int
        """
        coef = 1.0 if monotonic else 0.5
        scores = coef * eps / sensitivity * q
        probas = np.exp(scores - logsumexp(scores))
        return self.prng.choice(q.size, p=probas)

    def store(self, path: Path) -> None:
        """
        Store the MST mechanism state to a file.

        :param path: The path to store the state.
        :type path: Path
        """
        joblib.dump(
            {
                "epsilon": self.epsilon,
                "delta": self.delta,
                "compress": self._compress,
                "compressor": self.compressor,
                "cliques": self.cliques,
                "rho": self.rho,
                "sigma": self.sigma,
                "n_iters": self.n_iters,
                "_domain": self._domain,
                "max_model_size": self.max_model_size,
                "model_size": self.model_size,
                "fit_state": self.fit_state,
            },
            path / "state.joblib",
        )

        if self.model is not None:
            self.model.save(path / "estimator.pickle")

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Load the MST mechanism state from a file.

        :param path: The path to load the state.
        :type path: Path
        :return: The loaded MST mechanism.
        :rtype: MST
        """
        state = joblib.load(path / "state.joblib")
        obj = cls(
            epsilon=state["epsilon"],
            delta=state["delta"],
            n_iters=state["n_iters"],
            compress=state["compress"],
            domain=state["_domain"],
            max_model_size=state["max_model_size"],
        )
        obj.cliques = state["cliques"]
        obj.fit_state = state["fit_state"]
        obj.compressor = state["compressor"]
        obj.model_size = state["model_size"]
        model_path = path / "estimator.pickle"
        if model_path.exists():
            obj.model = GraphicalModel.load(model_path)
        return obj


class MSTGM(GraphicalGenerativeModel):
    """
    Maximum Spanning Tree (MST) mechanism is a differentially private generative model relying
    on selecting an optimal set of marginals to approximate the joint distribution of the data.
    It uses the exponential mechanism to select higher-order marginals based on their weights.
    The marginals are measured using the Laplace mechanism.
    The measured marginals are then used to estimate a maximum spanning tree which will be able to generate data.

    Ref: https://arxiv.org/pdf/2108.04978

    :param epsilon: Privacy budget.
    :type epsilon: float
    :param delta: Privacy parameter.
    :type delta: float
    :param n_iters: Number of iterations for inference.
    :type n_iters: int
    :param compress: Whether to compress the data.
    :type compress: bool
    :param max_model_size: Maximum model size in MB.
    :type max_model_size: int, optional
    :param domain: The domain of the data.
    :type domain: Domain, optional
    :param random_state: Random state for reproducibility.
    :type random_state: RandomState, optional
    """

    name = "mst"

    def __init__(
        self,
        epsilon: float = 1,
        delta: float = 1e-5,
        n_iters: int = 5000,
        compress: bool = True,
        max_model_size: Optional[int] = None,
        domain: Optional[Domain] = None,
        random_state: Optional[RandomState] = None,
        n_jobs: int = -1,
    ):
        super().__init__(domain=domain, random_state=random_state)

        self.epsilon = epsilon
        self.delta = delta

        self.generator = MST(
            epsilon=self.epsilon,
            delta=self.delta,
            n_iters=n_iters,
            compress=compress,
            domain=domain,
            prng=random_state,
            max_model_size=max_model_size,
            n_jobs=n_jobs,
        )

    def set_random_state(self, random_state: RandomState) -> None:
        """
        Set the random state for reproducibility.

        :param random_state: The random state.
        :type random_state: RandomState
        """
        super().set_random_state(random_state)
        self.generator.set_random_state(random_state)

    def set_domain(self, domain: Dict) -> None:
        """
        Set the domain of the data.

        :param domain: The domain.
        :type domain: Dict
        """
        super().set_domain(domain=domain)
        self.generator.set_domain(domain=domain)

    @to_path
    def store(self, path: Path) -> None:
        """
        Store the MSTGM model state to a file.

        :param path: The path to store the state.
        :type path: Path
        """
        super().store(path)
        self.generator.store(path)

    @classmethod
    @to_path
    def load(cls, path: Path) -> Self:
        """
        Load the MSTGM model state from a file.

        :param path: The path to load the state.
        :type path: Path
        :return: The loaded MSTGM model.
        :rtype: MSTGM
        """
        generator = MST.load(path)
        obj = cls(epsilon=generator.epsilon, delta=generator.delta)
        del obj.generator
        obj.generator = generator

        return obj
